// SPDX-License-Identifier: Apache-2.0
// Direct Metal compute runtime. No MLX, MPS, PyTorch or other tensor framework.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>
#include <array>

struct Runtime {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;
    id<MTLCommandBuffer> command;
    id<MTLComputeCommandEncoder> encoder;
    double lastGPUSeconds;
    id<MTLCounterSampleBuffer> samples;
    uint32_t sampleCount;
    std::vector<double> profileSeconds;
    uint64_t profileCPUStart, profileGPUStart;

};
struct Buffer { id<MTLBuffer> metal; };

// Plans own pipelines and integer binding slots, never activation/weight buffers.
// Each replay binds the current workspace; no GPU allocation survives via a plan.
struct PlanOp {
    id<MTLComputePipelineState> pipeline;
    std::array<uint32_t, 8> slots;
    std::array<uint8_t, 64> params;
    uint32_t count, group;
    uint64_t threads;
};
struct Plan {
    Runtime *owner;
    std::vector<uint64_t> sizes;
    std::vector<PlanOp> ops;
};

extern "C" {
void *mi_create(const char *source) {
    @autoreleasepool {
        auto r = new Runtime{};
        r->device = MTLCreateSystemDefaultDevice();
        if (!r->device) { delete r; return nullptr; }
        r->queue = [r->device newCommandQueue];
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.fastMathEnabled = NO;
        NSError *error = nil;
        r->library = [r->device newLibraryWithSource:[NSString stringWithUTF8String:source]
                                           options:options error:&error];
        // Do not log compiler diagnostics or paths into application stderr.
        if (!r->queue || !r->library) { delete r; return nullptr; }
        r->pipelines = [NSMutableDictionary new];
        return r;
    }
}

void mi_destroy(void *runtime) {
    @autoreleasepool { delete static_cast<Runtime *>(runtime); }
}

void *mi_buffer(void *runtime, const void *bytes, uint64_t size) {
    @autoreleasepool {
        auto r = static_cast<Runtime *>(runtime);
        if (!size || size > r->device.maxBufferLength) return nullptr;
        id<MTLBuffer> metal = bytes
            ? [r->device newBufferWithBytes:bytes length:size options:MTLResourceStorageModeShared]
            : [r->device newBufferWithLength:size options:MTLResourceStorageModeShared];
        if (!metal) return nullptr;
        return new Buffer{metal};
    }
}

void mi_free(void *buffer) {
    @autoreleasepool { delete static_cast<Buffer *>(buffer); }
}

int mi_read(void *buffer, void *output, uint64_t bytes) {
    auto b = static_cast<Buffer *>(buffer);
    if (bytes > b->metal.length) return 1;
    std::memcpy(output, b->metal.contents, bytes);
    return 0;
}

int mi_begin(void *runtime) {
    @autoreleasepool {
        auto r = static_cast<Runtime *>(runtime);
        if (r->command) return 1;
        r->command = [r->queue commandBuffer];
        r->sampleCount = 0;
        r->profileSeconds.clear();
        if (r->samples) {
            MTLTimestamp cpu, gpu;
            [r->device sampleTimestamps:&cpu gpuTimestamp:&gpu];
            r->profileCPUStart = cpu;
            r->profileGPUStart = gpu;
        } else r->encoder = [r->command computeCommandEncoder];
        if (!r->command || (!r->samples && !r->encoder)) {
            r->command = nil;
            return 1;
        }
        return 0;
    }
}

int mi_dispatch(void *runtime, const char *name, void **buffers, uint32_t count,
                const void *parameters, uint32_t size, uint64_t threads, uint32_t group) {
    @autoreleasepool {
        auto r = static_cast<Runtime *>(runtime);
        if (!r->command || (!r->samples && !r->encoder) || count > 8 || size != 64 || !threads || !group) return 1;
        NSString *key = [NSString stringWithUTF8String:name];
        id<MTLComputePipelineState> pipeline = r->pipelines[key];
        if (!pipeline) {
            id<MTLFunction> function = [r->library newFunctionWithName:key];
            if (!function) return 1;
            NSError *error = nil;
            pipeline = [r->device newComputePipelineStateWithFunction:function error:&error];
            if (!pipeline) return 1;
            r->pipelines[key] = pipeline;
        }
        if (group > pipeline.maxTotalThreadsPerThreadgroup) return 1;
        if (r->samples) {
            if (r->sampleCount + 2 > 4096) return 1;
            MTLComputePassDescriptor *pass = [MTLComputePassDescriptor computePassDescriptor];
            pass.sampleBufferAttachments[0].sampleBuffer = r->samples;
            pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = r->sampleCount;
            pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = r->sampleCount + 1;
            r->encoder = [r->command computeCommandEncoderWithDescriptor:pass];
            if (!r->encoder) return 1;
            r->sampleCount += 2;
        }
        [r->encoder setComputePipelineState:pipeline];
        for (uint32_t i = 0; i < count; ++i) {
            auto b = static_cast<Buffer *>(buffers[i]);
            [r->encoder setBuffer:b->metal offset:0 atIndex:i];
        }
        [r->encoder setBytes:parameters length:size atIndex:8];
        [r->encoder dispatchThreads:MTLSizeMake(threads, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
        if (r->samples) { [r->encoder endEncoding]; r->encoder = nil; }
        return 0;
    }
}

void *mi_plan_create(void *runtime, void **buffers, uint32_t count) {
    @autoreleasepool {
        auto r = static_cast<Runtime *>(runtime);
        if (!r->command || r->samples || !count || count > 2048) return nullptr;
        auto p = new Plan{r, {}, {}};
        for (uint32_t i = 0; i < count; ++i) {
            if (!buffers[i]) { delete p; return nullptr; }
            p->sizes.push_back(static_cast<Buffer *>(buffers[i])->metal.length);
        }
        return p;
    }
}

int mi_plan_add(void *plan, const char *name, const uint32_t *slots, uint32_t count,
                const void *params, uint64_t threads, uint32_t group) {
    @autoreleasepool {
        auto p = static_cast<Plan *>(plan);
        auto r = p->owner;
        if (!r->command || r->samples || !count || count > 8 || !threads || !group
            || p->ops.size() >= 2048) return 1;
        PlanOp op{};
        op.pipeline = r->pipelines[[NSString stringWithUTF8String:name]];
        if (!op.pipeline || group > op.pipeline.maxTotalThreadsPerThreadgroup) return 1;
        for (uint32_t i = 0; i < count; ++i) {
            if (slots[i] >= p->sizes.size()) return 1;
            op.slots[i] = slots[i];
        }
        std::memcpy(op.params.data(), params, 64);
        op.count = count; op.group = group; op.threads = threads;
        p->ops.push_back(op);
        return 0;
    }
}

uint64_t mi_plan_bytes(void *plan) {
    auto p = static_cast<Plan *>(plan);
    return sizeof(Plan) + p->sizes.capacity()*sizeof(uint64_t)
        + p->ops.capacity()*sizeof(PlanOp);
}

void mi_plan_free(void *plan) {
    @autoreleasepool { delete static_cast<Plan *>(plan); }
}

int mi_plan_run(void *runtime, void *plan, void **buffers, uint32_t count) {
    @autoreleasepool {
        auto r = static_cast<Runtime *>(runtime);
        auto p = static_cast<Plan *>(plan);
        if (p->owner != r || !r->command || !r->encoder || r->samples
            || count != p->sizes.size() || p->ops.empty()) return 1;
        // Validate every slot before encoding any work.
        for (uint32_t i = 0; i < count; ++i) {
            if (!buffers[i] || static_cast<Buffer *>(buffers[i])->metal.length != p->sizes[i])
                return 1;
        }
        for (const auto &op : p->ops) {
            [r->encoder setComputePipelineState:op.pipeline];
            for (uint32_t i = 0; i < op.count; ++i) {
                auto b = static_cast<Buffer *>(buffers[op.slots[i]]);
                [r->encoder setBuffer:b->metal offset:0 atIndex:i];
            }
            [r->encoder setBytes:op.params.data() length:64 atIndex:8];
            [r->encoder dispatchThreads:MTLSizeMake(op.threads, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(op.group, 1, 1)];
        }
        return 0;
    }
}

int mi_finish(void *runtime) {
    @autoreleasepool {
        auto r = static_cast<Runtime *>(runtime);
        if (!r->command || (!r->samples && !r->encoder)) return 1;
        [r->encoder endEncoding];
        [r->command commit];
        [r->command waitUntilCompleted];
        int failed = r->command.status == MTLCommandBufferStatusCompleted ? 0 : 1;
        double start = r->command.GPUStartTime, end = r->command.GPUEndTime;
        r->lastGPUSeconds = !failed && start > 0 && end >= start ? end - start : -1;
        if (r->samples && !failed && r->sampleCount) {
            MTLTimestamp cpu, gpu;
            [r->device sampleTimestamps:&cpu gpuTimestamp:&gpu];
            // Metal's paired CPU timestamps are nanoseconds, not mach absolute ticks.
            double secondsPerTick = gpu > r->profileGPUStart && cpu > r->profileCPUStart
                ? double(cpu-r->profileCPUStart)*1e-9 /
                    double(gpu-r->profileGPUStart) : 0;
            NSData *resolved = [r->samples resolveCounterRange:NSMakeRange(0, r->sampleCount)];
            if (!resolved || resolved.length != r->sampleCount*sizeof(MTLCounterResultTimestamp)
                || secondsPerTick <= 0) failed = 1;
            else {
                auto stamps = static_cast<const MTLCounterResultTimestamp *>(resolved.bytes);
                for (uint32_t i = 0; i < r->sampleCount; i += 2) {
                    uint64_t a = stamps[i].timestamp, b = stamps[i+1].timestamp;
                    if (!a || a == MTLCounterErrorValue || b == MTLCounterErrorValue || b < a) {
                        failed = 1; break;
                    }
                    r->profileSeconds.push_back(double(b-a)*secondsPerTick);
                }
            }
        }
        if (failed) r->profileSeconds.clear();
        r->encoder = nil;
        r->command = nil;
        return failed;
    }
}

// Opt-in profiling changes encoder boundaries; never enabled by inference defaults.
int mi_profile_enable(void *runtime, int enabled) {
    @autoreleasepool {
        auto r = static_cast<Runtime *>(runtime);
        if (r->command) return 1;
        r->profileSeconds.clear();
        r->sampleCount = 0;
        if (!enabled) { r->samples = nil; return 0; }
        if (![r->device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) return 1;
        for (id<MTLCounterSet> counters in r->device.counterSets) {
            if (![counters.name isEqualToString:MTLCommonCounterSetTimestamp]) continue;
            MTLCounterSampleBufferDescriptor *desc = [MTLCounterSampleBufferDescriptor new];
            desc.counterSet = counters;
            desc.storageMode = MTLStorageModeShared;
            desc.sampleCount = 4096;
            NSError *error = nil;
            r->samples = [r->device newCounterSampleBufferWithDescriptor:desc error:&error];
            return r->samples ? 0 : 1;
        }
        return 1;
    }
}

int mi_profile_read(void *runtime, double *seconds, uint32_t count) {
    auto r = static_cast<Runtime *>(runtime);
    if (r->command || count != r->profileSeconds.size()) return 1;
    if (count) std::memcpy(seconds, r->profileSeconds.data(), count*sizeof(double));
    return 0;
}

double mi_gpu_seconds(void *runtime) {
    return static_cast<Runtime *>(runtime)->lastGPUSeconds;
}

void mi_abort(void *runtime) {
    @autoreleasepool {
        auto r = static_cast<Runtime *>(runtime);
        if (r->encoder) [r->encoder endEncoding];
        r->encoder = nil;
        r->command = nil; // Uncommitted work is discarded.
    }
}
}
