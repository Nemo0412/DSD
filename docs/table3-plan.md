# dsd-real：从「管道跑通」到小规模 Table 3（Vanilla SD / SpecDec++ / EAGLE3）

## 背景

目标是在真实系统上小规模复现 paper 的 Table 3：分别给 Vanilla SD、SpecDec++、EAGLE3 测 TTFT、TPOT 和吞吐（req/s）。先做 n 个 draft 对 1 个 target，再扩到 n:m。真机数字要和 dsdSim（github.com/yfzzzyyls/dsdSim）尽量对得上。

### 我们现在到哪了（分支 `stage0-transport-layer`）

| 阶段 | 状态 |
|---|---|
| Stage 0：vLLM 0.27.1 的 hook，custom_class 的 `RemoteDraftProposer` | ✅ 在 L40S 上跑过；replay 输出与 baseline 逐 token 相同 |
| Stage 1：传输层抽象（InProcess / ZMQ / HTTP） | ✅ 三种都过了 parity 测试。HTTP 只在 loopback 上跑过，还没上 GPU |
| Stage 2a：fused，真 draft 模型，fp32 无损 | ✅ Qwen3-1.7B/0.6B，γ∈{1,2,4,8}，逐 token 相同 |
| Stage 2b：分布式 ZMQ，RTT 扫描 | ✅ TPOT 随 RTT 线性上升（R²=0.98，L̄=2.88） |
| Stage 3：N 个 draft → 1 个 target | 🟡 刚起步。2 个 drafter 的 RR 扇出能跑通。`router/ metrics/ workload/ experiments/ config/` 还是空包 |

现有代码和目标之间的差距：
- `proposer.py` 里 γ 写死为 `self.k`，不能按请求设 γ。
- 没有 request id，请求身份靠 slot 位置来认。
- 实验走的是离线 `LLM.generate`，请求没有到达时间，所以测不了 TTFT。
- 只支持一个 target。

### dsdSim 这边的情况（已对照仓库代码核实）

- **仓库里的东西复现不出 Table 3。** 没有 Table 3 的配置，也没有结果。VIDUR 延迟 profile（`thirdparty/Sai_speculative_vidur`、`data/vidur/cache`）不在仓库里，配置引用的 acceptance joblib 也不在，只有 `llama3_1b_vs_8b_500.joblib` 和两个 llama2 的 joblib。**结论：我们自己定义一张小 Table 3，sim 跑同样的配置来对比，不追 paper 上的绝对数字。**
- **横轴 node scale** 指的是 drafter 数量。`fused_vs_distributed_sweep.py` 里 target 固定是 20 个。我们的小规模版本取 N ∈ {1,2,4,7} 个 drafter 对 1 个 target：8 张 L40S，一张给 target，剩 7 张。N=8 就得让两个 drafter 共用一张卡（T2 档）。
- **三种算法在 sim 里怎么建模**（`src/sim.py`）：
  - Vanilla：`ConstantGammaPolicy(4)`，γ 恒为 4。
  - SpecDec++：`SpecPPGammaPolicy`（sim.py:137）。每个对话**只选一次** γ，参数 min 2、max 8、stop_threshold 0.7，依据是每个位置的接受概率。
  - EAGLE：drafter 生成几条 beam 分支，target 一条条串行 verify。这不是真正的 EAGLE3。
- **sim 的指标口径**（`Metrics.summary`，sim.py:893），和 vLLM 的不一样：
  - TTFT 截止到第一个至少接受 1 个 token 的 verify 轮。
  - TPOT = 每轮 RTT / 本轮 accepted，**按 token 汇总平均**，不是先算每个请求再平均。
  - **不计 bonus token。**
  - 吞吐 = 完成数 / (最晚完成 − 最早创建)，没有 burn-in。
- **workload 和导出格式：**
  - trace 文件是 `traces/gsm8k_trace_*.jsonl`，字段 `arrival_ms, prompt_tokens, target_tokens`。负载用 `per_draft_rps` 设定，所以 N 越大总负载越大。
  - 真机结果的导出格式按 `src/verification/validate.py::load_actual_run`，字段 `request_id, arrival_ms, prompt_tokens, completion_tokens, total_latency_ms, ttft_ms, draft_id`。

### 已确认的决定
- 模型对：**Llama-3.1-8B-Instruct（target）+ Llama-3.2-1B-Instruct（draft）**，和 sim 里现成的 `llama3_1b_vs_8b_500.joblib` 对得上。
- EAGLE3：**远程 EAGLE3 head**。target 把 aux hidden states 通过网络发给 drafter，drafter 跑 EAGLE3 head。

---

## 进度（2026-09-27）— 全部完成，结果见 [table3-results.md](table3-results.md)

| 步骤 | 代码 | 本地验证（WSL, 5070 Ti, Qwen3 小模型） | remote 门禁（NYULAB, Llama 8B/1B） |
|---|---|---|---|
| A driver+指标+workload | ✅ `experiments/driver.py` `metrics/{records,collector,export}.py` `workload/trace.py` | ✅ ZMQ 分布式 N=1 跑通，记账全过 | ✅（附说明）fp32 在线 40 请求：39 逐 token 相同，1 个在第 101 token 分叉，参考 logit top-2 差 1e-4（相对 4e-6）= fp32 近似平局，非算法问题；bf16 N=1 记账全过，TTFT 93ms / TPOT 16.4ms / L̄=3.85 / 传输开销 0.9ms |
| B Vanilla n:1 | ✅ `scripts/run_nto1.sh` `scripts/table3_sweep.sh` `scripts/make_table3.py` | ✅ | ✅ N∈{1,2,4,7}，见 [table3-results.md](table3-results.md) |
| C SpecDec++ | ✅ `policy/window.py`（逐行移植 sim 的 SpecPPGammaPolicy）+ `scripts/build_acceptance_profile.py` | ✅ γ 按请求选 {5,6,8}，记账全过 | ✅ γ∈{4,8}，N∈{1,2,4,7} |
| D EAGLE3 远程 head | ✅ K8 `target/hidden_states.py`；`draft/eagle3_model.py` `draft/engine_eagle3.py`；`--kind eagle3` | ✅ K8 对齐 cos≥0.999（错一位 ≤0.74）；接受长度 远程 1.970 vs vLLM 原生 1.973 | ✅ Llama-3.1-8B + yuhuili EAGLE3：接受长度 远程 2.98494 vs 原生 2.98501（相对差 2e-5）；fp32 无损 30/30 |
| E n:m | ✅ `router/{target_server,frontend}.py` + target 级 RR/Random/JSQ + `scripts/run_nm.sh` | ✅ 2 target + 1 drafter，JSQ 30/30 | ✅ 2 target : 2 drafter，RR vs JSQ，400 请求，记账全过 |
| E sim 侧 | ✅ `sim/`（shim + 标定拟合）+ `scripts/final_compare.sh` | ✅ | ✅ real vs sim：MAPE 19.5%（中位 11.1%）；未饱和区 TPOT/吞吐吻合，饱和区 sim 严重低估 |

实现中顺带修掉/确认的点：
- proposer 按**内容**认请求（vLLM 的 condense/swap_states 会挪行，新请求可能比旧占用者更长），不再按行号。
- drafter 按批 `generate()`（server 一次取完一步的全部消息），不再每条序列一次 engine 调用。
- 最后一轮 verify 可能超出 max_tokens：runner 的确认长度计入、输出被截断，之后那轮 draft 的 delta ≤ 0，按幽灵轮丢弃。
- 所有方法统一 `--chat-template`（Instruct 模型；EAGLE3 head 在裸文本上接受长度 1.6，套模板后 2.0），Llama 模板日期固定为 26 Jul 2024。
- 本地 WSL 需要 `VLLM_USE_V2_MODEL_RUNNER=0`（V2 需要 UVA）；driver/draft server 默认都设了，baseline 与 spec 同一个 runner。
- remote 是驱动 570（CUDA 12.8），装的是 vLLM 0.27.1 `+cu129` wheel（`scripts/env_nyulab.sh`）。
- **prefix caching 与远程 EAGLE3 不兼容**：cache 命中的位置不做前向、就没有 hidden states（GPU 上 Llama 模板 30 token 前缀命中 1 个 block 时由 K8 活断言当场抓到）。EAGLE3 模式强制关闭；为可比，Table 3 三种方法与门禁统一 `--no-prefix-caching`，并在 runner 初始化时拒绝这种组合。
- **重复 prompt**：Llama 模板头占 35 token，trace 里 30–35 token 的 prompt 只剩 1 个正文 token，400 个里 22 个撞成相同 prompt；相同 prompt 的 greedy 历史完全相同，proposer 按内容认请求就分不清，记账串号（EAGLE3 N=7 的 1/400 记账失败）。现在 prompt 保证两两不同，模板下正文至少 16 token（113/400 的 prompt 因此比 trace 长；导出的是真实长度，sim 侧也用真实长度）。
- 无损门禁的判据收紧为「逐 token 相同，或每个分叉点都是经 fp32 参考前向确认的近似平局（top-2 logit 差 < 1e-3）」：在线负载下 baseline 与 spec 的 batch 组成不同，fp32 归约顺序也随之不同（`scripts/inspect_divergence.py`）。

## 计划（按顺序做，每一步有门禁）

### Step A：online driver + 指标 + workload（后面所有行都依赖它）
- `dsdreal/workload/trace.py`：
  - 读 dsdSim 的 trace jsonl，按 `prompt_tokens` 配上真实 prompt。从 gsm8k 里按长度挑题目就够了。
  - 按 `per_draft_rps × N` 生成到达时间。
- `dsdreal/experiments/driver.py`：
  - 用进程内的 `LLMEngine`（`VLLM_ENABLE_V1_MULTIPROCESSING=0`），自己跑 `add_request` / `step()` 循环，每个请求在它的到达时间提交，`arrival_ts` 取预定到达时刻。
  - 不用 `AsyncLLM`：它总把 engine core 放到另一个进程，proposer 靠 `configure()` 设置的模块全局（client、记录钩子）在那个进程里不存在，而且两边时钟也不是同一个。
  - 每次 `step()` 返回后给新 token 打时间戳（和 transport 同一个 `perf_counter` 时钟）。
- **request id：** `SpeculateRequest` 加一个可选字段 `req_key`。
  - proposer 仍然按 slot 位置认请求，collector 用 prompt 前缀来关联：第一次 `propose` 带的 `prompt_token_ids` 就是 driver 那边的 prompt。
  - 加钩子 `configure(round_sink=...)`，每轮记录 `(seq_id, gamma, accepted, t_send, t_recv, draft_ms)`。
  - 这在已有的 `_stats_sink` 和 `SpeculateReply.sent_ms/recv_ms/draft_ms` 上扩展就行。
- `dsdreal/metrics/collector.py` + `export.py`：
  - **两套口径并列导出**：sim 口径（不含 bonus、TTFT 到首个有接受的轮、TPOT 按 token 汇总、完成数/时间跨度）和标准口径。
  - 同时导出 `validate.py::load_actual_run` 的格式，直接能喂给 dsdSim 的 `compare_runs`。
- **门禁：** Llama-8B/1B 上跑 N=1 Vanilla。按 token 汇总的 TPOT 要和每轮 accepted 记账自洽（Σaccepted + 轮数 == 输出长度），fp32 仍然无损。

### Step B：Vanilla SD 这一行（n:1）
- `run_draft_server.py` / `engine_vllm.py` 换成 Llama-3.2-1B。target 8B 用 bf16，fp32 只在测无损时用。
- 扫 N ∈ {1,2,4,7} 个 drafter：γ=4，`per_draft_rps` 固定，RR 路由（`policy/routing.py`），`netem/sleep.py` 注入 RTT（比如单向 5 ms）。
- 加一个 slurm sbatch，在一台 8 卡节点上起 N 个 drafter server 加 target driver，照 `slurm_ndrafter_fanout.sbatch` 的写法。
- **门禁：** 拿到小 Table 3 的 Vanilla 行。随 N 增大，吞吐应先上升后饱和，趋势和 sim 一致。

### Step C：SpecDec++ 这一行
- `dsdreal/policy/window.py`：移植 `SpecPPGammaPolicy`，语义保持一致：每个对话只选一次 γ，min 2、max 8、阈值 0.7。
  - 每个位置的接受概率取自 Step B 标定运行的逐位置统计。sim 是从 regressor 里取，对齐的是同一种东西。
- 改 proposer：
  - `num_speculative_tokens=8` 作为上限。
  - 新请求到来时选一次 γ，按 slot 存下来，写进 `SpeculateRequest.gamma`。
  - K1 已经验证过变长列表可以用。
- **门禁：** 每轮的 γ 落在 [2,8] 且和 policy 输出一致，输出仍然无损。

### Step D：EAGLE3 这一行（远程 head），最难，先做探针

按 context.md 的拆法：① runner 直接迁移 vLLM 的 eagle proposer（权重加载、d2t 映射、共享 target embed/lm_head 都复用）；② target→draft 发 hidden state 的协议；③ draft 侧 KV 回退 + 用新确认位置的 hidden state 做 reprefill（首轮是 prompt 全部位置）；④ 外面套现有的 `DraftServer`/`HttpDraftServer`，engine 实现同样的 `speculate`/`free` 接口即可，和 indep draft 读同样的 req；⑤ target 侧拿 hidden state（下面的 K8，最复杂，先做）。
- **先做 K8 探针：** 在 custom_class 路径下，target 能不能导出 aux hidden states？
  - EAGLE3 需要 target 的 3 个中间层，由 `use_aux_hidden_state_outputs` 控制。
  - `propose()` 的入参里没有 hidden states，所以要在插件里 patch `GPUModelRunner`，把本步的 aux hidden states 取出来交给 proposer。
  - 用现有 plugin.py 的 entry point 方式来做，并照 K7 的样子加 live 断言。
  - **探针不过就停下来，决定要不要改走 fork vLLM 打 patch。**
- **协议：**
  - `SpeculateRequest` 加字段 `hidden_states: bytes`，装自上一轮以来新确认的那些位置。
  - 体量：3×4096 bf16 ≈ 每 token 24 KB，ZMQ multipart 直接传原始字节就行。
- **drafter** `draft/engine_eagle3.py`：
  - 用纯 torch/HF 跑 `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` head，需要 target 的 embed、lm_head 权重和 d2t 词表映射。
  - 链式 draft γ 个 token，不做树，因为 vLLM 的 target 只能 verify 一条链。
  - drafter 自己维护 KV 状态，并回滚到 `num_confirmed`。
- **门禁：** fp32 无损；接受长度和同数据上 vLLM 原生 `method=eagle3` 接近。原生 eagle3 只当参照。

### Step E：n:m + 跑 sim 对照
- `dsdreal/router/frontend.py`：起 M 个 target 进程，router 在请求准入时选 target，策略有 RR 和 JSQ（JSQ 读 vLLM 的真实队列深度）。设计见 plan.md §3。
- **sim 侧：** 写一个和我们配置对应的 dsdSim 配置（1 个 target、N 个 drafter，RTT 和 trace 都相同）。
  - VIDUR 缺失，延迟改用我们实测的数，做成查表或常数延迟 provider 接进去。
  - acceptance 用 `llama3_1b_vs_8b_500.joblib`。
  - 最后出一张 Real vs Sim 对照表，带相对误差。

## 关键文件
- 修改：`dsdreal/target/proposer.py`、`dsdreal/proto/messages.py`、`dsdreal/target/plugin.py`（EAGLE3 的 hook）、`dsdreal/draft/engine_vllm.py`、`scripts/run_draft_server.py`
- 新增：`dsdreal/{workload/trace.py, experiments/driver.py, metrics/collector.py, metrics/export.py, policy/window.py, draft/engine_eagle3.py, router/frontend.py}`、`scripts/slurm_table3_*.sbatch`
- 复用：`policy/routing.py`（RR）、`netem/sleep.py`、`proto/transport.py::make_client`、`ZmqDraftClient` 的扇出逻辑

## 验证方式
- `pytest tests -q`（纯 CPU，必须保持全绿）。新增测试覆盖指标口径、trace 解析、SpecPP policy（用手算值对照 sim 公式）。
- GPU：每一步都重跑 fp32 无损（γ∈{1,2,4,8}），再跑这一步门禁对应的脚本（`scripts/verify_*.py` + sbatch）。
- 最终产物：`runs/table3_small/{vanilla,specpp,eagle3}_N{1,2,4,7}.json`，加一张和 paper 格式相同的 Markdown 表，与 sim 的数并排。

## remote 约束（context.md）
- 机器 NYULAB（直连 ssh，不是 slurm），**只挑没人用的卡，最多 3 张**。
- 所以先验 **2:1**（2 drafter + 1 target，3 卡）；N∈{1,2,4,7} 的扫描在 3 卡内用 T2（多个 drafter 共卡）或等更多卡。
- 优先做 vanilla 和 SpecDec++，EAGLE3 放后面。

## 算力分工（本地写代码 + 跑 CPU 测试，GPU 部分都丢 remote 的 slurm）

8B 模型 bf16 光权重就要约 16 GB，本地 5070 Ti（16 GB）装不下，所以凡是涉及模型的都在 remote 跑。

| 步骤 | 本地（无 GPU） | remote |
|---|---|---|
| A | driver、trace、collector、export 的代码和单测 | 1 张 L40S：8B+1B 跑 N=1 门禁和 fp32 无损，约 0.5 h |
| B | sbatch 脚本 | 整节点 8 卡：N∈{1,2,4,7} 扫描，约 1–2 h |
| C | SpecPP policy 和单测 | 2 卡：门禁；加上 N 扫描要 8 卡，约 1–2 h |
| D | 协议字段和 engine 的代码骨架 | 1–2 卡：K8 探针、EAGLE3 调通，**调试次数最多**，要反复提交 |
| E | router 和 sim 配置（sim 纯 CPU，本地跑） | M=2 需要 2 target + 6 drafter，整节点 |

跑的方式和以前一样：本地写好 `scripts/verify_*.py` + `slurm_*.sbatch`，你在 remote 上 `sbatch`，结果写到 `runs/`，再把 log 和 json 拿回来给我看。

## 建议先做
Step A：AsyncLLM online driver 和两套指标口径。每一行都依赖它，没有到达时间就测不了 TTFT 和吞吐。
