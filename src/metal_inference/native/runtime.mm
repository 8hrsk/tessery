// SPDX-License-Identifier: Apache-2.0
// Direct Metal compute runtime. No MLX, MPS, PyTorch or other tensor framework.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>
#include <cstring>
#include <mutex>

struct Runtime {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;
    id<MTLCommandBuffer> command;
    id<MTLComputeCommandEncoder> encoder;
};
struct Buffer { id<MTLBuffer> metal; };

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
        r->encoder = [r->command computeCommandEncoder];
        return r->encoder ? 0 : 1;
    }
}

int mi_dispatch(void *runtime, const char *name, void **buffers, uint32_t count,
                const void *parameters, uint32_t size, uint64_t threads, uint32_t group) {
    @autoreleasepool {
        auto r = static_cast<Runtime *>(runtime);
        if (!r->encoder || count > 8 || size != 64 || !threads || !group) return 1;
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
        [r->encoder setComputePipelineState:pipeline];
        for (uint32_t i = 0; i < count; ++i) {
            auto b = static_cast<Buffer *>(buffers[i]);
            [r->encoder setBuffer:b->metal offset:0 atIndex:i];
        }
        [r->encoder setBytes:parameters length:size atIndex:8];
        [r->encoder dispatchThreads:MTLSizeMake(threads, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
        return 0;
    }
}

int mi_finish(void *runtime) {
    @autoreleasepool {
        auto r = static_cast<Runtime *>(runtime);
        if (!r->command || !r->encoder) return 1;
        [r->encoder endEncoding];
        [r->command commit];
        [r->command waitUntilCompleted];
        int failed = r->command.status == MTLCommandBufferStatusCompleted ? 0 : 1;
        r->encoder = nil;
        r->command = nil;
        return failed;
    }
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
