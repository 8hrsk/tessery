# Qwen attention traversal experiment: not selected

Date: 2026-09-12. Baseline `e49b39c0966aceb3be7a9ac32ef90333c6339809`.
Isolated branch `perf/qwen-attention-next`, worktree
`/private/tmp/tessery-attention-next`. Production dispatch remains unchanged.

## Hypothesis and implementation

Group four adjacent eight-query tiles for one pair of Q-heads sharing a KV-head,
then advance to the next KV-head. This places eight related threadgroups next
to each other in the dispatch grid instead of only the original two Q-heads.
It attempts to improve K/V cache locality across threadgroups. Actual scheduling
and cache reuse are not guaranteed, and no hardware cache counters were collected.

The candidate changes only the mapping from linear threadgroup index to head and
query block. It retains the same 8×32 attention tile, 128 threads, shared arrays,
barriers, online-softmax updates, F32 operations and output addresses. The harness
guard is Qwen heads 16/KV8, head dimension128, causal attention, and physical sequence
width 128–512 divisible by 32. Tails, short queries and BGE keep their original route.
No new weights or model files were downloaded.

## Correctness

- 12 CPU topology tests passed: every batch/query/head is visited exactly once and
 every eight-group cluster shares a KV-head across four neighboring query blocks.
- 8 real Metal tests passed with Shader Validation: sequence 128/160/256/512,
 ragged lengths33/seq-1, causal and bidirectional masks. Output buffers are freshly
 initialized with NaNs separately for both kernels so missing writes cannot
 inherit baseline values. All outputs are finite and exactly equal to baseline.
- The independent F64 oracle retained `atol=2e-6,rtol=2e-5`.
- Full-model screen used two fixed model instances with native plans enabled;
 the candidate route was installed before capture. Timed calls confirmed plan
 hits and exactly 28 candidate attention dispatches per selected bucket. Every
 embedding exactly matched baseline.

## Measured screen

Apple M1,existing local Qwen3-Embedding-0.6B-4bit-DWQ model. Micro measurements
used 18 samples per label, balanced six permutations, two identical baseline labels,
and eight dispatches per command. Each route warmed for at least 0.3 seconds and
three calls. Synthetic inputs were fixed per case. Timings below use GPU command
timestamps,divided by eight; raw reports also retain wall timings.

| Batch / physical sequence | Baseline/candidate GPU median ratio | Identical control |
|---|---:|---|
|1×128|1.0278|passed|
|1×160|1.0190|passed|
|1×256|1.2144|failed — ratio is not reliable|
|1×512|1.0252|passed|
|2×512,ragged|0.9997|passed|

Because the reliable signal was small and absent for the larger batch,only one
bounded full-API screen was run: 6 samples per label, balanced permutations, at least
one second warmup per route. It is a screening run, not performance qualification.

| Logical lengths | Baseline/candidate full `encode()` median ratio |
|---|---:|
|`[24]`,unchanged control|0.9782|
|`[128]`|1.0010|
|`[159]`|1.0066|
|`[160]`|1.0017|
|`[256]`|1.0035|
|`[512]`|0.9722|
|`[159,160]`|0.9955|
|`[127,256]`|0.9853|
|`[3,7,10]`,unchanged control|1.0101|

All full-API identical-control screens passed, but the unchanged controls show
model-instance/timing variation larger than most apparent gains. There is no
sufficient evidence of useful end-to-end improvement, and the long512 screen is
worse. Do not interpret six samples as establishing an exact regression. A second
expensive run was intentionally omitted: the candidate is not selected this cycle.
Further acceptance would require more blocks, a fresh reverse run and reversed
model allocation order. No such benefit is claimed here.

Raw ignored evidence:

- `artifacts/attention-kvgroup-micro-first.json`
- `artifacts/attention-kvgroup-api-screen.json`

Reports contain source/harness hashes, input identity, raw timings and exact-output
checks; the model report also contains batch plans, vectors and runtime counters.
The experimental shader and harnesses remain on this branch only.
