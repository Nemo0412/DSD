# DSD

This repository holds the discrete-event simulator and the vLLM end-to-end system in one tree. Simulator documentation is first. The end-to-end system follows.

# dsdSim — Distributed Speculative Decoding Simulator

A discrete-event simulator for **distributed speculative decoding (DSD)** in datacenter environments, built on [SimPy](https://simpy.readthedocs.io/). Models draft–target LLM inference pipelines with realistic network topologies, scheduling policies, and acceptance rate prediction.

> **Terminology:** This simulator uses **draft** and **target** (not "edge" and "cloud") to reflect a pure-datacenter deployment where both models colocate in the same cluster.

---

## What It Simulates

- **Speculative decoding** pipeline: draft servers propose token chunks; target servers verify and accept/reject
- **Network-aware routing**: latency between draft and target servers based on physical topology
- **Scheduling policies**: JSQ, Random, Round-Robin, JSQ(d), Weighted JSQ, JIQ, and more
- **Acceptance modeling**:
  - **Replay mode (preferred):** per-request `acceptance_seq` from hardware profiling (`scripts/export_acceptance_seq_from_details.py`)
  - **Predictor fallback:** ML regressors (`.joblib`) or fixed acceptance rates when no sequence is present
- **Performance modeling**: per-token compute latency via default provider or VIDUR integration

---

## GPU Profiling Handoff for SpecDec++ and EAGLE

Use the same model pair, tokenizer, dataset split, temperature, top-p, maximum output length, and random seeds for profiling and simulation. Store these settings in each trace row's `metadata` so results remain attributable to one decoding configuration.

### SpecDec++ / linear speculative decoding

Record the per-request verification outcome as `acceptance_seq`, a flat list of `0`/`1` values in draft-token order. Within each speculation round, the accepted length is the longest leading run of ones, terminated by the first zero.

If the GPU profiler emits `speculative_profiler.py --details-jsonl` records, convert them directly:

```bash
python scripts/export_acceptance_seq_from_details.py \
  --details results/specdecpp_details.jsonl \
  --workload traces/workload.jsonl \
  --output traces/specdecpp_hardware_acceptance.jsonl
```

At minimum, retain `request_id`, prompt and target lengths, `acceptance_seq`, model pair, dataset, sampling settings, speculation-window settings, and seed.

### EAGLE / tree speculative decoding

Linear `acceptance_seq` is insufficient for EAGLE. Record one object per verification round:

```json
{
  "request_id": "req-0001",
  "tree_accept": [
    {
      "round": 0,
      "depth": 4,
      "candidates": 16,
      "verified": 16,
      "accepted_path_len": 3,
      "accepted_branch": [0, 1, 0],
      "draft_generation_ms": 2.31,
      "target_verification_ms": 5.84
    }
  ],
  "metadata": {
    "algorithm": "EAGLE",
    "dataset": "GSM8K",
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 121
  }
}
```

Also retain the tree topology or parent index for every submitted candidate when it cannot be reconstructed from `accepted_branch`.

With `framework: eagle`, a request that carries `tree_accept` replays it one entry per verification round (`src/trace/tree_accept.py`). Each round advances the output by `accepted_path_len + 1` (the accepted path plus the target's token). The target verifies `tree_nodes` candidates in one batch job; when that field is missing, `verified` and then `candidates` are used. Draft generation and target verification latencies still come from the calibrated performance models. Once a request's log runs out, the controller falls back to `acceptance_seq` or the acceptance predictor.

The EAGLE controller also models the hidden-state traffic and draft-side initialization of distributed EAGLE. These settings live under `speculation.eagle`:

| Key | Default | Effect |
|---|---|---|
| `hidden_dim` | 4096 | Target hidden-state width |
| `hidden_state_bytes_per_value` | 2.0 | Bytes per hidden-state value |
| `tree_index_bytes_per_node` | 4.0 | Tree-structure bytes sent per candidate node |
| `feature_init_base_ms`, `feature_init_ms_per_token` | 0.0 | Draft feature initialization over the prompt, charged once before the first tree |

The prefill reply carries the prompt's hidden states. Each verify reply carries the hidden states of the committed tokens. These bytes only change latency when the network has a bandwidth (`network.bandwidth_mbps`).

### Acceptance sensitivity re-simulation

The sensitivity runner now exports mean accepted length, TPOT, token throughput, rounds/request, candidates verified/request, draft generation, target queueing, target verification, and forward/response network time:

```bash
python scripts/run_acceptance_sensitivity_sim.py \
  --config experiments/configs/acceptance_sensitivity_64.yaml \
  --traces-dir traces/perturbed \
  --out-dir experiments/results/acceptance_sensitivity
```

Burst perturbation is position-preserving: shortening converts surplus accepted bits to rejects in place; it never deletes bits or randomly pads a sequence.

---

## Recent Changes

### Fat-tree Network Topology (`src/network/`)

Added a three-level fat-tree network model reflecting realistic datacenter GPU cluster topology.

**Hierarchy:**
```
Root switch
  └─ Pod switch  ×  pods_per_root
       └─ Node   ×  nodes_per_pod
            └─ GPU  ×  gpus_per_node  (connected via NVLink)
```

**One-way latency rules:**

| Path | Latency |
|---|---|
| Same node (NVLink) | 60 ns |
| Same pod, different node | 2 × 150 = 300 ns |
| Different pod | 2 × 150 + 2 × 250 = 800 ns |

**Configuration** (`src/network/fat_tree_config.json`):
```json
{
  "pods_per_root": 4,
  "nodes_per_pod": 4,
  "gpus_per_node": 4,
  "latency": {
    "nvlink_ns": 60,
    "node_to_pod_ns": 150,
    "pod_to_root_ns": 250
  }
}
```
All parameters (topology size and per-layer latency) are editable in this JSON file without touching code.

**GPU index assignment:** each device is assigned a global GPU index (0-based, row-major across pod → node → GPU). Given two GPU indices, the simulator computes the correct one-way latency automatically.

**Usage in YAML config:**
```yaml
connectivity:
  network_model:
    type: fat_tree
    # optional: override any parameter inline
    # gpus_per_node: 8
    # latency:
    #   nvlink_ns: 60
    # optional: explicit GPU placement for benchmarks
    # draft_gpu_indices: [0, 4, 16, 20]
    # target_gpu_indices: [8, 24, 32, 48]
```

**New files:**
- `src/network/fat_tree.py` — `FatTreeTopology` class with `latency_ns(gpu_i, gpu_j)` and `latency_ms(gpu_i, gpu_j)`
- `src/network/fat_tree_config.json` — editable topology config

**Modified files:**
- `src/network/topology.py` — added `fat_tree` type dispatch (no NetworkX dependency)
- `src/network/__init__.py` — exports `FatTreeTopology`

---

### LAB (Length-Aware Batching) — `src/sim.py`

Added a new `queue_policy: "lab"` to the `PhaseScheduler`. LAB groups requests by context-length bucket so that jobs with similar sequence lengths are co-scheduled into the same batch, reducing padding waste.

**How it works:**
- Sort key becomes `(priority, context_len // bucket_size, arrival_seq)` instead of `(priority, arrival_seq)`
- The `PriorityStore` naturally co-schedules same-bucket jobs without any extra collection logic
- Configurable `lab_bucket_size` (tokens); defaults to 64 when policy is `"lab"`

**Config:**
```yaml
scheduler:
  decode:
    queue_policy: lab
    lab_bucket_size: 64   # group requests within ±64 token context windows
```

**Benchmark** (`experiments/benchmark_batching.py`) — routing fixed at JSQ, 90 req/s, answer length 30–600 tokens (wide distribution to stress LAB):

| Policy | TPOT avg | TTFT avg | Throughput | Completed convs |
|---|---|---|---|---|
| FIFO | 1.237 ms | 15.49 ms | 1993 jobs/s | 298 |
| **LAB-64** | **1.230 ms** ✓ | **14.95 ms** ✓ | 1995 jobs/s | 307 |
| LAB-128 | 1.240 ms | 15.31 ms | **2025 jobs/s** ✓ | 309 |

**Key findings:**
- **LAB-64** reduces TPOT by 0.5% and **TTFT by 3.5%** vs FIFO — short-context requests finish faster, giving users earlier first tokens
- **LAB-128** (coarser buckets) maximises throughput (+1.6%) by enabling larger, more uniform batches
- Fine-grained bucketing (64) favours latency; coarse bucketing (128) favours throughput
- In real GPU serving, benefits would be larger since actual computation scales with the *maximum* sequence length per batch (not the average used here)

**`src/performance/factory.py` — Lazy VIDUR import**

The VIDUR provider was imported unconditionally at module load time, causing a `FileNotFoundError` even when `performance_model.type: default` was configured. Fixed with lazy import: VIDUR is only loaded when explicitly requested.

**`src/sim.py` — Performance metadata hoisting**

`prefill_latency_per_token` and `decode_latency_per_token` specified in a tier's `metadata` block (or directly on the tier) are now hoisted to the top level of the device entry dict, making them visible to `DefaultPerformanceProvider`.

---

### JSQ vs Random Benchmark (`experiments/benchmark_jsq_vs_random.py`)

Compares **JSQ** (Join Shortest Queue) and **Random** routing policies on a fat-tree datacenter topology with randomly distributed draft and target GPUs.

**Run:**
```bash
# From anywhere in the repo
python experiments/benchmark_jsq_vs_random.py
```

**Sample results (12 drafts → 4 targets, LOAD=90 req/s, seed=2025):**

GPU placement spans all three latency tiers (60 / 300 / 800 ns), target utilization ~112% (overloaded).

| Router | TPOT avg | TPOT p95 | Throughput | RTT avg | Completed convs |
|---|---|---|---|---|---|
| **JSQ** | **1.260 ms** ✓ | 2.801 ms | 1306 jobs/s | **3.153 ms** ✓ | 305 |
| Random | 1.297 ms | 2.802 ms | **1316 jobs/s** | 3.223 ms | 315 |
| Round-Robin | 1.316 ms | 2.802 ms | 1312 jobs/s | 3.268 ms | 320 |

**Key findings:**
- **JSQ** achieves the lowest TPOT (−2.9% vs Random, −4.2% vs Round-Robin) and lowest RTT under overload, because it avoids routing to already-queued targets
- **Round-Robin** has the highest TPOT and RTT — it ignores queue state entirely and can repeatedly route to the same overloaded target
- All three routers achieve similar throughput when targets are overloaded (load-shedding behavior dominates)
- JSQ's advantage grows with load — at light load (~50% utilization) differences are <1%

---

## Repository Structure

```
dsdSim/
├── src/
│   ├── sim.py                        # Main simulator (SimPy discrete-event)
│   ├── network/
│   │   ├── fat_tree.py               # Fat-tree topology (NEW)
│   │   ├── fat_tree_config.json      # Editable topology parameters (NEW)
│   │   ├── topology.py               # Latency lookup (clos, complete, fat_tree)
│   │   └── fabric.py                 # SimPy link fabric (bandwidth, jitter, queuing)
│   ├── performance/
│   │   ├── factory.py                # Provider factory (lazy VIDUR import)
│   │   └── default_provider.py       # Per-token latency fallback
│   ├── acceptance/                   # ML acceptance regressors (.joblib)
│   └── trace/                        # Trace loading (JSONL)
├── experiments/
│   ├── benchmark_jsq_vs_random.py    # JSQ vs Random on fat-tree (NEW)
│   ├── benchmark_batching.py         # FIFO vs LAB batching comparison (NEW)
│   └── configs/
│       └── fat_tree_test.yaml        # Minimal smoke-test config (NEW)
├── docs/
│   └── DESIGN.md                     # Full design document
└── requirements.txt
```

---

## Quick Start

```bash
# Install dependencies
pip install simpy pyyaml networkx

# Run smoke test (fat-tree + JSQ, 300ms sim)
cd src
python sim.py --config ../experiments/configs/fat_tree_test.yaml

# Run JSQ vs Random benchmark
python ../experiments/benchmark_jsq_vs_random.py
```

---

## Original AWS Trainium Implementation

The `aws-trn/` directory contains the original gRPC-based distributed speculative decoding implementation for AWS Trainium (trn1) instances using LLaMA 3.2-1B (draft) and LLaMA 3.1-8B (target). See `aws-trn/CLAUDE.md` for details.



This repository supports two architectures for **speculative decoding** on AWS Trainium, using **Meta LLaMA 3.2-1B** (draft) and **LLaMA 3.1-8B** (target) models:

1. **Distributed Architecture**: Client-side draft model with multiple round-trips
2. **Fused Architecture**: Server-side fused model with single round-trip

## Project Structure

Below is an overview of the repository structure and how the modules relate to each other:

```
choral-spec-internal/
├── main.py                      # CLI entry point; parses args and launches roles (draft, target, fused_target, fused_client, verify)
├── inference/                   # Package for model loading, speculative decoding, and verification logic
│   ├── model_loader.py          # Utilities to load/compile models, includes load_fused_speculative_model()
│   ├── draft_worker.py          # Distributed: Draft client with speculative decoding
│   ├── target_worker.py         # Distributed: Target server (token-by-token verification)
│   ├── fused_draft_worker.py    # Fused: Simple client that sends requests
│   ├── fused_target_worker.py   # Fused: Server with FusedSpeculativeDecoder
│   ├── speculative.py           # Distributed speculative decoding algorithm
│   └── verify.py                # Verification utilities for standalone model testing
├── grpc_comm/                   # gRPC definitions and generated code
│   ├── inference.proto          # Protocol definitions for both architectures
│   ├── inference_pb2.py         # Generated Python classes
│   └── inference_pb2_grpc.py    # Generated gRPC client/server code
├── compare_architectures.py     # Automated performance comparison script
├── requirements.txt             # Python dependencies
├── README.md                    # Documentation
└── CLAUDE.md                    # Instructions for future Claude instances

```

## Dpendencies

Create a Trainium instance with AWS Neuron SDK using EC2 with the following settings:

1. 1. **Name:** Your Name
   2. **AMI:** Deep Learning AMI Neuron (Ubuntu 22.04)
   3. **Instance type:** trn1.2xlarge
   4. **Key pair (login):** create a new key pair
   5. **Metadata version [under “Advanced details”]:** V2 only (otherwise, you will encounter a not authorized error)
   6. **When connecting to these instances via SSH, use the username of *ubuntu***
2. Activate the Neuron virtual environment to run inference by running `source /opt/aws_neuronx_venv_pytorch_2_5_nxd_inference/bin/activate`.

Install dependencies

```
pip install grpcio==1.71.0 grpcio-tools==1.66.2
pip install gevent
pip install --upgrade transformers-neuronx
```

## Setup

1. **Clone Repo & Install**:

   ```
   git clone https://github.com/yfzzzyyls/choral-spec
   ```
2. **Download Models** (1B draft, 3B target) from Hugging Face. For example:

   ```
   cd ~
   mkdir models
   huggingface-cli download --token YOURTOKEN meta-llama/Llama-3.2-1B --local-dir /home/ubuntu/models/llama-3.2-1b
   ```
3. **Optinal: Generate new grpc files**

   ```
   cd grpc_comm
   python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. inference.proto
   ```

   Notice: in the newly generated inference_pb2_grpc.py, if you have the following code:

   ```
   import inference_pb2 as inference__pb2
   ```

   replace it with:

   ```
   from . import inference_pb2 as inference__pb2
   ```

## **Usage:**

### **Optional:**

Clean cache before compile:

```
rm -r /var/tmp/neuron-compile-cache
```

### **Distributed Architecture**

#### **1. Start the Target Model Server**

```
python main.py --role target --model /home/ubuntu/models/llama-3.1-8b/ --port 50051 --sequence_length 128 --batch 2 --profile --top_p 1.0
```

#### **2. Run the Draft Model Client**

```
python main.py --role draft --model /home/ubuntu/models/llama-3.2-1b/ --target_host 18.189.128.139 --port 50051 --prompt_text prompt.txt --sequence_length 128 --profile --top_p 1.0 --temperature 0.9
```

### **Fused Architecture**

#### **1. Start the Fused Speculative Server**

```
python main.py --role fused_target --draft_model /home/ubuntu/models/llama-3.2-1b --target_model /home/ubuntu/models/llama-3.1-8b --port 50051 --sequence_length 128 --batch 2 --tp_degree 2 --speculation_length 5
```

#### **2. Run the Fused Client**

```
python main.py --role fused_client --target_host 18.189.128.139 --port 50051 --prompt_text prompt.txt --temperature 0.9 --top_p 1.0 --speculation_length 5 --profile
```

### **Performance Comparison**

Run automated comparison between both architectures:

```
python compare_architectures.py --prompt-file prompt.txt --num-runs 3
```

Note: By default, both architectures will generate tokens up to the maximum sequence length (128) for fair comparison.

### **Example Output**

#### Distributed Architecture:
```
2025-07-26 23:04:15,748 INFO inference.draft_worker: [Thread 0] completed in 1.41s, tokens=48, throughput=34.15 t/s

=== Final Outputs (CONCURRENT) ===
[Prompt 0 Output]:
The history of AI can be traced back to 1960s, when the AI pioneers started to develop symbolic AI, also named as the von Neumann program. The early programs developed in the 1950s, including the Logic Theorist, a program that can prove various
```

#### Fused Architecture:
```
2025-07-27 00:45:28,733 INFO inference.fused_draft_worker: Server generation time: 2582.84ms
2025-07-27 00:45:28,733 INFO inference.fused_draft_worker: Tokens per second: 19.36
2025-07-27 00:45:28,733 INFO inference.fused_draft_worker: Acceptance rate: 71.07%

[Prompt 0 Output]:
The history of AI can be traced back to the 1950s with the advent of the computer. The first application of AI was in the field of mathematics, where it was used to solve complex equations. Since then, AI has been used in a variety of fields, including finance, healthcare
```

## **Performance Profiling Stats**

```
INFO:inference.verify:Performance metrics saved to performance_target_only_20250408_013547.csv and performance_target_only_20250408_013547.json
```

Performance stats are saved to .cvs and .json files

## **Run a Single Model for Verification**

You can also run either the draft or target model **standalone** (without speculative decoding) to verify its generation output token-by-token. This is useful for debugging and sanity checks to ensure each model behaves as expected given a prompt.

To run the **target model** by itself on a prompt:

```
python main.py --role verify_target --model /home/ubuntu/models/llama-3.1-8b --prompt_text prompt.txt --sequence_length 128 --profile
```

This will load the 8B target model and generate tokens up to the sequence length, printing each generated token as it arrives, followed by the full output text.

Similarly, to run the **draft model** by itself:

```
python main.py --role verify_draft --model /home/ubuntu/models/llama-3.2-1b --prompt_text prompt.txt --sequence_length 128 --profile
```

This will use the 1B draft model to generate text token-by-token for the given prompt.

*Note:* In verification modes, the model will be compiled on the fly if a compiled Neuron model is not found. By default, **`--sequence_length 128`** is used; ensure you use the same sequence length that the model was compiled with (or specify `--sequence_length` accordingly) to avoid recompilation.

## **Key Parameters**

### Common Parameters:
- `--sequence_length`: Maximum sequence length (default: 128)
- `--max_new_tokens`: Maximum number of tokens to generate
- `--temperature`: Temperature for sampling (default: 1.0)
- `--top_p`: Top-p for nucleus sampling (default: 0.9)
- `--profile`: Enable performance profiling

### Distributed-Specific:
- `--gamma`: Number of draft tokens per verification step (default: 4)
- `--batch`: Batch size for model compilation (default: 2)

### Fused-Specific:
- `--speculation_length`: Number of tokens to speculate at once (default: 5)
- `--tp_degree`: Tensor parallelism degree (default: 2 for trn1.2xlarge)

## **Performance Notes**

- **Distributed architecture**: ~34 tokens/second throughput
- **Fused architecture**: ~19 tokens/second throughput
- Fused models always use `batch_size=1` internally
- Both architectures can use the same port (just not simultaneously)


---

# End-to-end system


分布式投机解码（disaggregated speculative decoding）的真实系统实现，用来端到端验证 DSD-Sim 的仿真结论。

路线是 **插 vLLM，不 fork**：draft 和 target 各跑一个独立的 vLLM 引擎，靠 vLLM 原生的
`speculative_config.method = "custom_class"` 把一个远程 drafter 塞进 target 的投机路径。
Stage 0 的七个探针在 pin 住的平台上全绿，Stage 2 的 fused 和 distributed 两条腿都已经在真 GPU 上验过。

平台：**vLLM 0.27.1 / 8×L40S (sm_89, 48GB)**，pin 见 [requirements/gpu.txt](requirements/gpu.txt)。

---

## 架构

```
   Draft vLLM Server                Draft Client                        Target vLLM Server
 ┌────────────────────┐      ┌──────────────────────┐              ┌──────────────────────────┐
 │ VllmDraftEngine    │      │ ┌─────┐┌────┐┌─────┐ │  ┌────────┐  │ ┌────────────┐ ┌───────┐ │
 │  1. Propose: γ tok │◄────►│ │ IPC ││ZMQ ││HTTP │ │◄►│Routing │◄►│ │Draft Runner│ │Target │ │
 │  2. Rollback: 按   │      │ └─────┘└────┘└─────┘ │  │ policy │  │ │ [custom_   │ │Runner │ │
 │     confirmed 前缀 │      │   一个 DraftClient    │  └────────┘  │ │  class]    │ │       │ │
 │     重跑 generate  │      │        契约           │  RoundRobin  │ └────────────┘ └───────┘ │
 └────────────────────┘      └──────────────────────┘              │   RemoteDraftProposer    │
   engine_vllm.py ✅          transport.py 三种都 ✅                 │   .propose() 每 step 一次 │
   engine_replay.py ✅        server.py / http_server.py ✅          └──────────────────────────┘
   engine_echo.py   ✅        policy/routing.py ✅                        proposer.py ✅
```

控制循环在 **target** 一侧：每个 engine step，vLLM 自己调用
`RemoteDraftProposer.propose()`，propose 里同步向各 drafter 扇出、等齐、返回逐请求变长的
draft token 列表。

**三条契约（由代码和测试固定，不是约定俗成）：**

- `propose()` 跑在 engine step 内部，**没有 event loop** → 整条 `DraftClient` 路径是阻塞的。
- `speculate()` 一次吃整个 batch，**不是**逐请求 API。墙钟必须是 `max(RTT_i)` 而不是
  `Σ RTT_i`；逐请求 API 会悄悄把 N 个 draft 节点串行化，正好毁掉要测的东西。
- accept/reject 反馈**不需要单独通道**：`propose()` 入参的 `num_tokens_no_spec[i]` 就是 slot i
  的确认长度，跟上一轮的差就是这一轮接受了几个（含 bonus token）。
  `configure(stats=[])` 把它直接导出来，不用从时序斜率反推。

请求身份目前靠 slot 位置恢复（`num_tokens_no_spec` 只会在同一请求占用该 slot 期间增长），
因为协议层还没有 request id。

---

## 传输层：三种实现，一个契约

`context.md` 里写的「从 zmq 到 http，先包装成抽象层」就是这一层。三种实现都在
[proto/transport.py](dsdreal/proto/transport.py) 后面，`make_client(kind, ...)` 按名字选：

| | `InProcessDraftClient` | `ZmqDraftClient` | `HttpDraftClient` |
|---|---|---|---|
| 线 | 直接函数调用 | DEALER/ROUTER + msgpack | HTTP/1.1 + JSON |
| 服务端 | 无 | [`DraftServer`](dsdreal/draft/server.py) | [`HttpDraftServer`](dsdreal/draft/http_server.py) |
| 并发方式 | 不适用（串行，延迟按 max 记账） | 一个 DEALER 管道化多条消息，send-all → recv-all | 按 drafter 分组，**每 drafter 一个 POST**，线程池并发 |
| 每 step 往返数 | 0 | 每 drafter 1 次（管道化） | 每 drafter 1 次 |
| 错误传播 | 直接抛 | `ERR ` 帧 → 客户端 re-raise | HTTP 400 + `{"error"}` → 客户端 re-raise |
| 状态 | ✅ Stage 2a GPU 验过 | ✅ Stage 2b GPU 验过 | ✅ loopback 测试通过，**未上 GPU** |

**这不是"二选一"，是两条都留着。** 判据是 `SpeculateReply.net_excess_ms`
（= 实测 RTT − 注入延迟 − drafter 计算），也就是序列化 + 排队的真实成本：
HTTP 便于调试（curl 就能打、抓包能读、换语言不用改协议），ZMQ 开销更低。
两边 loopback 上都断言 `net_excess_ms < 20ms`，真链路上哪个更划算要用这个数说话，不是靠假设。

[tests/test_transport_parity.py](tests/test_transport_parity.py) 把这件事钉死：同一串
propose/rollback 回合分别走三种传输，**token 必须逐个相同** —— 传输只能改变成本，不能改变结果。

### HTTP 侧两个不是可选项的设置

不设的话测到的是自己的配置错误，不是网络：

- **`protocol_version = "HTTP/1.1"`** —— stdlib 默认 1.0，每个响应后关连接，于是**每个 step 都付一次
  TCP 握手**，白白多一个 RTT，还会记进 `net_excess_ms`。
- **`disable_nagle_algorithm = True`** —— 否则几十字节的小响应要等 40ms ACK 定时器。
  客户端侧 `http.client` 自己已经设了 `TCP_NODELAY`，服务端侧没有。

还有一个是结构性的：**一个 POST 带一个 drafter 这一步的全部请求**。ZMQ 靠一个 DEALER
管道化多条消息拿到并发，HTTP/1.1 没有可用的 pipelining，逐请求 POST 会把同一 drafter 的
N 个序列串行成 N 次往返。按 drafter 打包后，往返数只和 drafter 数有关，和 batch 大小无关 ——
这条由服务端计数器 `speculate_posts` 断言，**不是**用计时断言：loopback 上八次多余往返不到 1ms，
计时根本看不出来，只会在真链路上才暴露。

---

## 已验证（真 GPU）

| 阶段 | 内容 | 结果 |
|---|---|---|
| **Stage 0** | K1–K7 七个 hook 探针在 0.27.1 上重跑 | 全绿，`runs/stage0_probes.json` |
| **Stage 0** | `custom_class` + `ReplayDraftEngine` 端到端回放 | 输出与 greedy baseline **逐 token 相同** |
| **Stage 2a** | fused：真 draft 模型 + `InProcessDraftClient`，Qwen3-1.7B + Qwen3-0.6B，fp32，γ∈{1,2,4,8} | 全部**逐 token 等于**非投机 baseline |
| **Stage 2b** | distributed：同样两个模型，draft 在独立进程独立 GPU，真 ZMQ socket，扫 RTT | **TPOT 随 RTT 线性，R²=0.98，L̄=2.88** |
| **Stage 2b** | 接受长度 fused vs ZMQ 对拍 | 完全一致，`rel_diff_of_means=0.0` |
| **Stage 3 起步** | N-drafter 扇出，2 个真 ZMQ drafter + RoundRobin | 5/5 分流，token 与 baseline 完全一致 |

Stage 2b 的线性度是 K2「propose() 在关键路径上」的运行时复查 —— 跨两个真实进程和一条真
socket 之后它仍然成立。

脚本在 [scripts/](scripts/)（`verify_*.py`）+ 对应的 `slurm_*.sbatch`。

---

## drafter 的状态机与回滚

slide 上写的「rollback: prefix + verified + bonus 重 prefill」，落地成三步：

1. **丢掉上一轮的投机尾巴** —— 提议完 γ 个 token 后 drafter 比确认前缀多 γ 个位置。
2. **追加 target 的权威 token** —— 它**不等于**自己提议过的 token：最后一个是 target 的
   bonus / 纠正 token，drafter 从没见过。
3. **用绝对的 `num_confirmed` 校验**，不一致就**大声报错**而不是从错误前缀继续 draft。
   用绝对值而非增量，还让协议在重试下幂等（HTTP 客户端的 keep-alive 断线重试正靠这条）。

[`VllmDraftEngine`](dsdreal/draft/engine_vllm.py) 用的就是 slide 上那个办法：每轮对完整确认前缀
**无状态地重跑 `generate()`**，靠 `enable_prefix_caching`（公开且稳定的引擎 flag）让重复前缀便宜，
不碰任何 vLLM 内部。正确性完全建立在确定性上：fp32 + `enforce_eager` + greedy，
同一前缀必然产生同一续写，命不命中 prefix cache 都一样。

代价是每轮多一次 prefill —— 这不是忠实的 SD 回滚。更"高效"的做法（draft 侧常驻、可暂停、
能回滚的 session）要拿版本敏感的内部实现重写一遍 target 自己的 KV 回滚调度，而真 drafter
每次被拒还是得回到重新提交这条路上。

---

## Stage 0 的几条约束

**探针全绿 = 「插 vLLM 不 fork」成立。** 完整的 0.21.0 首轮记录见
[docs/stage0-findings.md](docs/stage0-findings.md)（注意：那份文档写的是最初在 WSL / RTX 5070 Ti /
vLLM 0.21.0 上的结果，平台已经换到 0.27.1 / L40S 并重跑，最新数据以
`runs/stage0_probes.json` 为准）。

换版本时踩到的两处 API 漂移，已在探针里修掉：`NgramProposer.propose()` 多了一个前置的
`num_speculative_tokens` 参数；`Scheduler.schedule()` 多了 `throttle_prefills`。

仍然成立的约束：

1. **`draft_probs` 恒为 `None`**（23/23）→ 是 **token-match 接受、T=0 无损**，不是 Leviathan
   随机拒绝采样。slide 上那句 "vLLM V1 sd support greedy only?" —— 在这条路径上是的。
2. **`RejectionSampler` 有第二个调用点**（显存 profiling 期的 `_dummy_sampler_run`，随机 logits）。
   L2 tap 必须有 `live` 门闩，否则 acceptance 数据被污染且**不报错**（788 vs 288）。
3. **0.27.1 原生支持 `custom_class`**，所以 0.21.0 时代设想的 `NgramProposer` 子类 shim 不需要了；
   `plugin.py` 里那个 `GPUModelRunner.__init__` monkeypatch 也已经删掉 —— 它会无条件覆盖
   `self.drafter`，把所有非 custom_class 的运行一起弄坏。

---

## 快速开始

```bash
source scripts/env.sh          # 任何入口都要先 source

# 1. 纯 CPU：传输层 + proposer + 路由的全部测试，无 GPU、无模型、秒级
pytest tests -q                # 97 passed (torch 可选：EAGLE3 引擎测试需要)

# 2. 需要 GPU
python scripts/verify_vllm_hooks.py            # Stage 0 七探针（--probe k1 可单跑）
python scripts/verify_replay_proposer.py       # custom_class 回放，端到端无损
python scripts/verify_fused_real_draft.py      # Stage 2a
python scripts/verify_distributed_rtt_sweep.py # Stage 2b（需先起 draft server）
sbatch scripts/slurm_fused_real_draft.sbatch   # 集群上批量跑
```

`requirements/base.txt` 刻意不含 torch/vLLM：传输层和它的测试**永远不该需要 GPU 栈**，
这个分界线就是靠那个文件守住的。GPU 依赖在 `requirements/gpu.txt` / `pip install -e .[gpu]`。

### 起一个 draft server

```bash
# ZMQ（Stage 2b 用的就是这条）
python scripts/run_draft_server.py --drafter-id d0 --bind tcp://0.0.0.0:5560
# HTTP（同一个引擎，换条线）
python scripts/run_draft_server.py --transport http --drafter-id d0 --bind http://0.0.0.0:5560
```

target 侧在构造 `LLM` **之前**把 client 交给 proposer：

```python
from dsdreal.proto.transport import make_client
from dsdreal.target import proposer

proposer.configure(make_client("http", addresses={"d0": "http://node1:5560"}))
#                              "zmq",  addresses={"d0": "tcp://node1:5560"}   # 换这一行即可

target = LLM(
    model=..., dtype="float32", enforce_eager=True,
    speculative_config={
        "method": "custom_class",
        "model": "dsdreal.target.proposer.RemoteDraftProposer",
        "num_speculative_tokens": GAMMA,
    },
)
```

`configure()` 吃任意 `DraftClient`，所以换传输不需要改 proposer 一行代码。

---

## 目录

```
dsdreal/
├── proto/      messages.py（msgspec，JSON 和 msgpack 都能出）
│               transport.py（DraftClient + InProcess / Zmq / Http 三种实现）
├── draft/      base.py  engine_echo.py（无 GPU）  engine_vllm.py（真模型，批量 generate）
│               eagle3_model.py  engine_eagle3.py（远程 EAGLE3 head）
│               engine_replay.py（回放）  server.py（ZMQ）  http_server.py（HTTP）
├── target/     proposer.py（custom_class）  hidden_states.py（K8）  plugin.py（entry point）
├── policy/     routing.py（drafter RR；target RR/Random/JSQ）  window.py（SpecDec++）
├── netem/      base.py  null.py  sleep.py（RTT/jitter/延迟矩阵）
├── router/     target_server.py  frontend.py（N:M）
├── metrics/    records.py  collector.py  export.py
├── workload/   trace.py        experiments/  driver.py
scripts/        env.sh  verify_*.py  slurm_*.sbatch  run_draft_server.py
tests/          test_transport.py  test_zmq_transport.py  test_http_transport.py
                test_transport_parity.py  test_remote_draft_proposer.py  test_routing.py
docs/           stage0-findings.md
plan.md         完整设计：架构、指标对齐、分阶段、终止判据
```

## Table 3（小规模）：Vanilla SD / SpecDec++ / EAGLE3 × N 个 drafter

计划与进度见 [docs/table3-plan.md](docs/table3-plan.md)，**结果（真机 Table 3、门禁、real vs sim、n:m）见 [docs/table3-results.md](docs/table3-results.md)**。在线实验的链路：

```
trace (dsdSim gsm8k, Poisson 到达) ─► experiments/driver.py ─► 进程内 LLMEngine（target）
                                         │  每 step propose() ─► N 个 draft server（ZMQ）
                                         └► metrics/collector.py ─► dsd_metrics.json / steps.jsonl / actual_run.jsonl
```

| 模块 | 作用 |
|---|---|
| [experiments/driver.py](dsdreal/experiments/driver.py) | 按到达时间 `add_request`、逐 step 打 token 时间戳（不用 AsyncLLM：它把 engine 放到别的进程，proposer 的模块全局配置就丢了） |
| [metrics/collector.py](dsdreal/metrics/collector.py) | 每轮 accepted 由下一轮的 `num_confirmed` 增量推出；**两套口径**并列：标准口径 vs dsdSim 口径（无 bonus、TTFT 到首个有接受的轮、TPOT 按 token 汇总） |
| [workload/trace.py](dsdreal/workload/trace.py) | dsdSim trace（`data/dsdsim/`，MIT）→ 长度精确的 prompt；`--chat-template` 保持总长不变 |
| [policy/window.py](dsdreal/policy/window.py) | SpecDec++：逐行移植 sim 的 `SpecPPGammaPolicy`，逐位置接受概率来自真机标定 |
| [target/hidden_states.py](dsdreal/target/hidden_states.py) | K8：让 custom_class proposer 拿到 target 的 EAGLE3 aux hidden states（按新确认位置切片） |
| [draft/eagle3_model.py](dsdreal/draft/eagle3_model.py) / [engine_eagle3.py](dsdreal/draft/engine_eagle3.py) | 远程 EAGLE3 head（移植自 vLLM llama_eagle3）：KV 回退 + reprefill + 链式 draft，批处理 |
| [router/](dsdreal/router/) | N:M：target server 进程 + router frontend（RR / Random / JSQ，TTFT 从 router 入口算） |
| [sim/](sim/) | 让 dsdSim 在同样配置下跑（缺 VIDUR 用标定文件替代），`calibrate_from_run.py` 从真机 run 拟合延迟 |

门禁（都已在本地小模型上过，remote 上用 Llama-3.1-8B / 3.2-1B 复跑）：

- **Step A**：在线负载下 fp32，spec 输出 == 无 draft baseline（`scripts/compare_outputs.py`）；记账 `Σ(accepted+1)` 与输出长度自洽。
- **Step C**：每请求 γ ∈ [2, 8] 且等于 policy 输出，仍无损。
- **Step D**：K8 对齐（与 HF 参考逐位置 cos ≥ 0.98，且明显高于错一位）；远程 EAGLE3 接受长度与 vLLM 原生 `method=eagle3` 相差 ≤ 5%（本地：1.970 vs 1.973）；fp32 无损（本地 30/30 通过）。

remote（NYULAB，只用空闲卡、最多 3 张）一键：`nohup scripts/run_all_nyulab.sh > runs/pipeline.log 2>&1 &`，
进度在 `runs/pipeline.status`，结果表在 `runs/table3/table3.md`。
