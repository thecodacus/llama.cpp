# llama.cpp — `perf` fork

Faster **mixture-of-experts (MoE) models on consumer GPUs** when the experts don't fit in VRAM.
This is [llama.cpp](https://github.com/ggml-org/llama.cpp) plus a handful of opt-in changes for the
`--n-cpu-moe` case (experts in system RAM, the rest on the GPU). Everything is **off by default**
and produces the **same tokens** as upstream.

| Feature | What it does | How to turn it on |
| --- | --- | --- |
| **MoE expert cache** | Keeps each layer's most-used experts resident in VRAM; the rest stay in RAM. With several GPUs, each GPU caches the experts of its own layers. | `--moe-cache-profile FILE --moe-cache-slots N` |
| **Routing profiles** | `llama-moe-trace` records which experts a model picks, with your real sampling settings, so the cache knows what to keep. | `llama-moe-trace` (see below) |
| **Async CPU splits** | CPU and GPU halves of each MoE layer run at the same time. | on by default with the cache (`--no-sched-async-cpu` to disable) |
| **Prefill speedups** | Pins offloaded expert memory and prefetches it on a second CUDA stream. | `GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_PREFETCH_EXPERTS=1` |
| **Qwen3.8-Flash-Next MTP** | `--spec-type draft-mtp` works for `qwen4exp` with its separate draft head. | `-md mtp-*.gguf --spec-type draft-mtp` |

Measured decode speed on one **RTX 3060 12 GB** (`-ngl 99 -ncmoe 99 -fa 1`):

| Model | Baseline | With the expert cache |
| --- | --- | --- |
| Qwen3.6-35B-A3B Q4_K_M (256 experts/layer) | 42.3 tok/s | **51.3 (+21%)** @ 124 slots |
| — same, plus `--spec-type draft-mtp` | 41.7 | **69.3 (+66%)** @ 112 slots |
| — same, plus async CPU splits (default on) | 41.7 | **74.2 (+78%)** @ 88 slots |
| GLM-4.7-Flash Q4_K_M (64 experts/layer) | 32.1 | **46.3 (+44%)** @ 40 slots |
| Laguna-S-2.1-118B-A8B IQ4_XS (256 experts/layer) | 11.5 | **12.1 (+5%)** @ 36 slots |
| Qwen3.8-Flash-Next UD-IQ3_XXS (512 experts/layer), with MTP | 16.6 | **24.4 (+47%)** @ 56 slots |

On **two RTX 3060s**, Qwen3.8-Flash-Next with the cache split across both cards runs **~40 tok/s**
(see [Two GPUs](#two-gpus)).

Cache-capable architectures: `qwen35moe`, `qwen4exp` (Qwen3.8-Flash-Next), `deepseek2`, `laguna`.
Other models run unchanged.

## Quick start

You need an NVIDIA GPU, the CUDA toolkit, CMake, and an MoE model in GGUF format.

**1. Build**

```bash
git clone --branch perf https://github.com/thecodacus/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server llama-moe-trace
```

**2. Run it once without the cache** so you have a baseline. `-ncmoe 99` puts every expert in RAM:

```bash
./build/bin/llama-server -m model.gguf -ngl 99 -ncmoe 99 -fa on -c 16384
```

Open http://localhost:8080, ask something, and note the tokens per second.

**3. Record a routing profile** (once per model):

```bash
MOE_TRACE_OUT=profile.csv ./build/bin/llama-moe-trace -m model.gguf \
  -ngl 99 -ncmoe 99 -fa on -c 4096 -n 512 --temp 0.7 \
  -p "Write a Python function that parses a CSV file and explain how it works."
```

**4. Serve with the cache.** Start around 64 slots:

```bash
./build/bin/llama-server -m model.gguf -ngl 99 -ncmoe 99 -fa on -c 16384 \
  --moe-cache-profile profile.csv --moe-cache-slots 64
```

The load log should show:

```
init_moe_expert_cache: expert cache: <layers> layers x 64 slots, <size> MiB uploaded to CUDA0
```

**5. Find your slot count.** Raise `--moe-cache-slots` until the load prints
`pack allocation failed`, step back down, then leave about 1 GB of VRAM free (`nvidia-smi`) for
long prompts. More slots means more experts in VRAM and faster decode.

That's it. Everything below is for squeezing out more.

## Recipes

### One GPU, with MTP speculative decoding

For models with an MTP head (Qwen3.6, Qwen3.8), stack it on top of the cache and drop a few slots
to make room:

```bash
./build/bin/llama-server -m model.gguf -ngl 99 -ncmoe 99 -fa on -c 16384 \
  --moe-cache-profile profile.csv --moe-cache-slots 56 \
  --spec-type draft-mtp --spec-draft-n-max 2
```

Qwen3.8-Flash-Next keeps its MTP head in a separate file: add `-md mtp-Qwen3.8-Flash-Next-*.gguf`.

### Two GPUs

Split layers with `-sm layer`; each GPU caches the experts of its own layers, so `-ts` decides how
the cache is shared. The MTP draft head (`-ngld 99`) lands on the last GPU, so give that GPU fewer
layers:

```bash
./build/bin/llama-server -m Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
  -ngl 99 -ncmoe 99 -fa on -c 65536 -ctk q8_0 -ctv q8_0 \
  -sm layer -ts 28,20 --moe-cache-profile profile.csv --moe-cache-slots 136 \
  -md mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf -ngld 99 --spec-type draft-mtp --spec-draft-n-max 2 \
  --load-mode mmap --lazy-mode on --no-op-offload
```

On 2× RTX 3060 12 GB this used 11.2 + 11.4 GB and ran **39.9 tok/s** (temp 0, warm server), vs
35.0 for the best layout without the cache. Tune `-ts` until both cards end up about equally full.

### Several models behind one server

Every flag works as a key in a `--models-preset` INI section:

```ini
[qwen38-flash]
model = /models/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
n-cpu-moe = 99
moe-cache-profile = /models/traces/profile.csv
moe-cache-slots = 136
```

The cache flags also read `LLAMA_ARG_MOE_CACHE_PROFILE` / `LLAMA_ARG_MOE_CACHE_SLOTS` (and the
legacy `GGML_MOE_CACHE_PROFILE` / `GGML_MOE_CACHE_SLOTS`, which `llama-bench` accepts too).

## Better routing profiles

The quick-start profile is enough to start. For the best hit rate, trace the kind of work you
actually do.

- **Merge contrasting workloads.** One trace per workload, then concatenate them:
  `cat code.csv chat.csv long.csv > profile.csv`. A merged profile measures within 1% of
  per-workload specialist profiles.
- **Trace with your server's sampling settings.** The tracer uses the normal sampling flags
  (`--temp`, `--top-k`, `--top-p`, `--min-p`, `--seed`, penalties); `--temp 0` is greedy.
- **Render prompts with the model's chat template.** `-p`/`-f` take raw text, so a plain question
  skips the template the server would apply. Let a running `llama-server` render it:

  ```bash
  curl -s localhost:8080/apply-template -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"<your prompt>"}],
         "chat_template_kwargs":{"enable_thinking":true}}' \
    | jq -r .prompt > review.txt

  MOE_TRACE_OUT=review.csv ./build/bin/llama-moe-trace -m model.gguf \
    -ngl 99 -ncmoe 99 -fa on -c 16384 -n 1024 \
    --temp 0.7 --top-p 0.95 --top-k 20 --min-p 0 --seed 101 -f review.txt
  ```

- **Layout doesn't matter while tracing.** Routing doesn't depend on `-sm`, `-ts` or `-ncmoe`, so
  trace with whatever layout loads.
- Only generated tokens count toward the profile; prompt rows are written with negative positions
  and skipped at load.

On Qwen3.8-Flash-Next (2× RTX 3060), a profile from eight 0.1k–15k-token prompts traced at temp 0.7
averaged the same as a greedy short-prompt profile (30.0 vs 29.7 tok/s with server sampling at
temp 1.0), trading ~3 tok/s on code for ~1–3 tok/s on scripts, reasoning and long prompts. Build the
profile from the work you want fastest.

## Tuning

- **`--moe-cache-slots` is the main knob** — experts cached per layer. Throughput rises with slot
  count until the pack no longer fits in VRAM. The pack is all-or-nothing per GPU: an oversized
  request logs `pack allocation failed on <device> - expert cache disabled for its N layers` and
  those layers run at baseline speed (it does not partially fill). The pack costs roughly
  slots × (cached layers on that GPU) × (one expert's gate+up+down bytes). Check the GPU's memory
  after load: a pack that fell back leaves it far below full.
- **Leave ~900 MB of VRAM free beyond the pack.** A slot count that loads can still crash on the
  first large prompt: runtime CUDA pool growth allocates beyond what the load-time check sees.
  Size slots against the biggest prompt you will serve, not against "it loaded".
- **On one GPU, fill VRAM to just under the ceiling and don't sweat the split.** Near the maximum,
  a marginal MB is worth about the same as cache slots or as fully-resident layers (lower
  `--n-cpu-moe`). Pure `-ncmoe 99` + max slots is the simple default; a hybrid (e.g. `-ncmoe 30` +
  fewer slots) buys ~1% decode and ~3% prefill at best.
- **Context size competes with the pack.** KV grows with `-c` and shrinks the viable slot count.
  Compressing the KV cache (`-ctk`/`-ctv`) frees VRAM that converts directly into slots — often
  worth more than the KV precision costs.
- **Speculative decoding stacks multiplicatively.** `--spec-type draft-mtp` composes with the
  cache (+48% cache × +12% MTP ≈ +66% on Qwen); reserve ~1 GB for the draft context by dropping
  a few slots.
- **The cold and hot chains overlap by default.** CPU graph splits run on a worker thread so the
  GPU hot chain runs concurrently with the CPU cold chain (`--no-sched-async-cpu` to disable;
  `llama-bench --sched-async-cpu 0,1` benches both). Worth +4-5% with speculative decoding, ~±2%
  without it; outputs stay bit-identical either way.
- **Expert size decides the payoff.** Big experts (GLM: ~5 MiB each) gain the most per slot;
  small experts need high slot counts before the win beats the dual-path overhead (~25% traffic
  coverage is roughly break-even). If VRAM only fits <15% of the expert count, expect single-digit
  gains (Laguna above).
- **Two GPUs: balance the split so both fill.** Filling both cards to the last few hundred MB was
  slower again in testing (144 vs 136 slots above); stop a little short.
- **Benchmark on a warm server.** The first requests after a load run while expert pages are still
  coming off disk; compare layouts over several rounds on a loaded server, not one request per
  fresh start.
- **Profiles are model-specific, workload-tolerant.** A wrong-workload profile still helps
  (+28% measured on Qwen worst-case) but loses about half the win; a merged profile recovers
  nearly all of it. Regenerate only if your usage changes character entirely.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `cannot open profile '...'` | Path not visible to the process (e.g. not mounted into the container). |
| `pack allocation failed on <device>` | Slot count too big for that GPU — reduce slots, or move layers off it with `-ts`. |
| `no CPU-resident MoE layers` | Experts are already on GPU (no `--n-cpu-moe`) — nothing to cache. |
| No init line, no warning | Architecture not wired for the cache — model runs unchanged. |
| Model loads, then context creation OOMs | Pack fits but KV/compute don't — drop a few slots or shrink/compress KV. |
| Crash on the first long prompt | Not enough free VRAM for the prompt's workspace — drop slots until ~1 GB stays free. |

## Prefill speedups for offloaded experts

Two environment variables speed up prompt processing when experts live in RAM, independent of the
cache. Both are off by default and token-identical.

| Env var | What it does |
| --- | --- |
| `GGML_CUDA_REGISTER_HOST=1` | Page-locks (pins) the mmap'd CPU expert weights so host→device copies go straight over DMA instead of through the driver's hidden bounce buffer (~6–7 → ~20 GB/s). |
| `GGML_SCHED_PREFETCH_EXPERTS=1` | Prefetches each layer's experts on a second CUDA stream, so the weight uploads overlap compute instead of stalling the GPU. |

On an RTX 3060 12 GB with Qwen3.6-35B-A3B (`--n-cpu-moe 26`), prefill at 2048 went from
**~1143 → ~1880 tok/s (+64%)**:

```bash
GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_PREFETCH_EXPERTS=1 \
./build/bin/llama-bench -m model.gguf -ngl 99 -ncmoe 26 -p 2048 -n 0 -r 5 -b 2048 -ub 2048
```

---

# Upstream llama.cpp

Everything below is the upstream README.

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>


## Recent API changes

- [Changelog for `libllama` API](https://github.com/ggml-org/llama.cpp/issues/9289)
- [Changelog for `llama-server` REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

## Hot topics

- **Hugging Face cache migration: models downloaded with `-hf` are now stored in the standard Hugging Face cache directory, enabling sharing with other HF tools.**
- **[guide : using the new WebUI of llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/16938)**
- [guide : running gpt-oss with llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/15396)
- [[FEEDBACK] Better packaging for llama.cpp to support downstream consumers 🤗](https://github.com/ggml-org/llama.cpp/discussions/15313)
- Support for the `gpt-oss` model with native MXFP4 format has been added | [PR](https://github.com/ggml-org/llama.cpp/pull/15091) | [Collaboration with NVIDIA](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss) | [Comment](https://github.com/ggml-org/llama.cpp/discussions/15095)
- Multimodal support arrived in `llama-server`: [#12898](https://github.com/ggml-org/llama.cpp/pull/12898) | [documentation](./docs/multimodal.md)
- VS Code extension for FIM completions: https://github.com/ggml-org/llama.vscode
- Vim/Neovim plugin for FIM completions: https://github.com/ggml-org/llama.vim
- Hugging Face Inference Endpoints now support GGUF out of the box! https://github.com/ggml-org/llama.cpp/discussions/9669
- Hugging Face GGUF editor: [discussion](https://github.com/ggml-org/llama.cpp/discussions/9268) | [tool](https://huggingface.co/spaces/CISCai/gguf-editor)
- WebGPU support is now available in the browser, see a blog/demo introducing it [here](https://reeselevine.github.io/llamas-on-the-web/).

----

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
