# 小规模 Table 3：真机结果与 dsdSim 对照

NYULAB（4× RTX A6000，只用空闲卡，最多 3 张），vLLM 0.27.1。计划与实现细节见 [table3-plan.md](table3-plan.md)；数据在 [results/table3_client/](../results/table3_client/)（主结果，端云场景）和 [results/table3_final/](../results/table3_final/)（数据中心场景）。

## 场景：按 dsdSim 的端云模型

dsdSim 把 drafter 建模成**用户侧设备**（README：「Client-side draft model」；DESIGN.md：手机/笔记本/边缘 GPU 做 draft，数据中心 GPU 做 target），请求在 drafter 的客户端产生：prompt 先上传到 target，结果再传回客户端。主结果按这个场景跑（driver `--client-side`）：请求在客户端时间到达，经上行 RTT/2 才进入 target，每个 token 经下行 RTT/2 才算到达用户。每轮的工作两种场景完全相同，所以 TPOT/吞吐不变，TTFT 与端到端多一个 RTT。控制循环仍在 target 的 vLLM step 里（`custom_class` 只能如此），时间上与「drafter 驱动」逐事件等价（每轮：draft → 上行 → verify → 下行）。

## 配置

| | |
|---|---|
| target | Llama-3.1-8B-Instruct，bf16，1 张 A6000，eager，max_num_seqs 64，prefix caching 关闭（远程 EAGLE3 需要，三种方法统一；dsdSim 本身也不建模 prefix cache） |
| draft | Vanilla / SpecDec++：Llama-3.2-1B-Instruct（独立 vLLM 进程）；EAGLE3：yuhuili/EAGLE3-LLaMA3.1-Instruct-8B 远程 head（target 把 aux hidden states 经 ZMQ 发过去） |
| N 个 drafter | target 独占 1 卡；drafter 轮转摆在另 2 张卡：N=1/2 每卡 ≤1 个（T1），N=4 每卡 2 个、N=7 每卡 3–4 个（T2，共卡）。每个 run 的实际摆放记在 `placement.json` |
| 网络 | 注入 RTT 10 ms（单向 5 ms，无抖动），ZMQ；传输层实测额外开销 ≈ 1 ms |
| 负载 | dsdSim `gsm8k_trace_master` 的长度，开环 Poisson，总速率 = 1 req/s × N，前 min(400, 120N) 个请求；chat 模板，prompt 两两不同，输出严格 `target_tokens`（ignore_eos），greedy |
| 方法 | Vanilla γ=4；SpecDec++ = dsdSim `SpecPPGammaPolicy` 逐行移植（min 2 / max 8 / 阈值 0.7，每请求选一次 γ，逐位置接受率用 γ=8 标定 run 实测，实际选出 γ∈{4, 8}）；EAGLE3 链式 γ=4 |
| 干净度 | 每个 run 前重新挑空闲卡、结束时检查卡上有无其他用户进程：全部 12 个 run `clean` |

## 门禁（全部通过）

| 门禁 | 结果 |
|---|---|
| Step A 无损（fp32，在线负载，spec vs 无 draft baseline） | 40/40 逐 token 相同；记账（每轮 accepted 由下一轮确认长度推出）全过 |
| Step D 远程 EAGLE3 vs vLLM 原生 `method=eagle3` | 平均接受长度 2.98494 vs 2.98501（相对差 2e-5） |
| Step D 无损（fp32） | 30/30 逐 token 相同 |
| K8 hidden-state 对齐（本地 Qwen3-1.7B，强制 chunked prefill） | 与 HF 参考逐位置 cos ≥ 0.999，错一位时 ≤ 0.74 |
| Table 3 全部 12 个 run | 记账全过 |

（早一轮 fp32 门禁出现过 1/40 在第 101 token 分叉；参考前向给出的 top-2 logit 差 1e-4（相对 4e-6），是在线负载下 batch 组成不同导致的 fp32 近似平局，门禁现在会自动做这项检查。）

## 真机 Table 3（端云场景）

**dsdSim 口径**（TTFT 到首个有接受 token 的轮结束，TPOT = Σ轮时间/Σaccepted（不含 bonus），吞吐 = 完成数/时间跨度）：

| Method | Metric | N=1 | N=2 | N=4 | N=7 |
|---|---|---|---|---|---|
| Vanilla SD | TTFT (ms) | 138.2 | 148.0 | 196.7 | 538.4 |
|  | TPOT (ms) | 19.4 | 20.3 | 28.2 | 65.3 |
|  | Thr. (req/s) | 0.86 | 1.86 | 3.68 | 5.93 |
|  | accept ratio | 0.640 | 0.640 | 0.642 | 0.640 |
| SpecDec++ | TTFT (ms) | 167.5 | 181.0 | 293.5 | 5134.9 |
|  | TPOT (ms) | 19.1 | 20.1 | 33.8 | 79.8 |
|  | Thr. (req/s) | 0.86 | 1.86 | 3.64 | 5.01 |
|  | accept ratio | 0.545 | 0.550 | 0.547 | 0.555 |
| EAGLE3 | TTFT (ms) | 125.0 | 129.0 | 172.5 | 363.9 |
|  | TPOT (ms) | 18.4 | 19.6 | 26.6 | 55.9 |
|  | Thr. (req/s) | 0.86 | 1.87 | 3.70 | 6.08 |
|  | accept ratio | 0.501 | 0.493 | 0.492 | 0.497 |

**标准口径**（TTFT = 首 token − 到达；TPOT = 每请求 (末−首)/(n−1) 的均值；吞吐去掉头尾 10%；平均接受长度含 bonus）：

| Method | Metric | N=1 | N=2 | N=4 | N=7 |
|---|---|---|---|---|---|
| Vanilla SD | TTFT (ms) | 100.8 | 108.1 | 149.5 | 425.8 |
|  | TPOT (ms) | 17.4 | 18.3 | 25.4 | 58.5 |
|  | Thr. (req/s) | 0.82 | 1.95 | 3.71 | 5.41 |
|  | mean accept len | 3.54 | 3.54 | 3.55 | 3.54 |
| SpecDec++ | TTFT (ms) | 125.4 | 137.8 | 228.8 | 4978.2 |
|  | TPOT (ms) | 18.5 | 19.7 | 33.1 | 77.7 |
|  | Thr. (req/s) | 0.82 | 1.95 | 3.56 | 4.40 |
|  | mean accept len | 4.27 | 4.25 | 4.19 | 4.23 |
| EAGLE3 | TTFT (ms) | 92.3 | 94.8 | 129.3 | 269.6 |
|  | TPOT (ms) | 16.6 | 17.8 | 24.4 | 51.0 |
|  | Thr. (req/s) | 0.82 | 1.96 | 3.74 | 5.54 |
|  | mean accept len | 3.01 | 2.97 | 2.97 | 2.99 |


读法：

- 吞吐在 N ≤ 4 时等于供给负载；N=7（7 req/s）超过 target 容量，三种方法都饱和，天花板 EAGLE3 6.08 > Vanilla 5.93 > SpecDec++ 5.01。
- 排序 **EAGLE3 > Vanilla > SpecDec++**。与 paper Table 3（SpecDec++ 优于 Vanilla）相反：按实测的逐位置接受率，SpecDec++ 规则对长 prompt 选到 γ=8，每步 draft 与 verify 都更贵；负载低时 TPOT 与 Vanilla 持平，接近饱和时先崩（N=7 TTFT 5.1 s，请求在 vLLM 的 `max_num_seqs` 准入队列里排队）。
- EAGLE3 每轮接受得比 1B drafter 少（L̄ 2.99 vs 3.54），但 head 只有一层、draft 便宜（每轮 ≈ 10 ms vs 1B 的 ≈ 24 ms），整体最快。

## 分卡部署：一个 step 的时间花在哪

每个 engine step 拆成：propose 墙钟（= 最慢 drafter 的 draft + 网络/传输）与 target 的 verify step（`scripts/deployment_report.py`，p50）：

| run | 摆放 | batch B | propose 墙钟 | draft（最慢） | 网络+传输 | verify step |
|---|---|---|---|---|---|---|
| vanilla N=1 | T1：d0@GPU1 | 2 | 34.2 | 23.2 | 11.0 | 25.9 |
| vanilla N=2 | T1：每卡 1 个 | 4 | 35.8 | 24.8 | 10.9 | 26.0 |
| vanilla N=4 | T2：每卡 2 个 | 13 | 55.0 | 43.8 | 11.1 | 29.5 |
| vanilla N=7 | T2：每卡 3–4 个 | 48 | 117.2 | 85.6 | 30.0 | 68.8 |
| specpp N=1 | T1 | 2 | 51.3 | 40.3 | 11.2 | 26.3 |
| specpp N=2 | T1 | 5 | 54.5 | 43.5 | 11.0 | 26.7 |
| specpp N=4 | T2 | 16 | 94.8 | 83.4 | 11.2 | 35.8 |
| specpp N=7 | T2 | 61 | 213.8 | 166.5 | 46.5 | 89.5 |
| eagle3 N=1 | T1 | 2 | 21.3 | 10.0 | 11.3 | 26.6 |
| eagle3 N=2 | T1 | 4 | 22.0 | 10.1 | 11.1 | 26.9 |
| eagle3 N=4 | T2 | 12 | 40.0 | 17.2 | 22.0 | 29.6 |
| eagle3 N=7 | T2 | 42 | 81.4 | 48.7 | 32.3 | 66.6 |

- **T1（每个 drafter 独占一卡）**：网络+传输 ≈ 11 ms（注入 10 ms + 传输 ≈ 1 ms），verify ≈ 26 ms 基本恒定，差别只在 draft（EAGLE3 10 ms / 1B 24 ms / SpecDec++ 42 ms）。
- **T2（共卡）**：draft 时间成倍上涨，「网络+传输」也涨到 22–47 ms——多出来的是请求在 drafter server 里排队、等同卡的其他 drafter 进程；同时 batch 变大，verify 升到 67–90 ms。N ≥ 4 的瓶颈主要是 3 卡限制下的 drafter 共卡争用。要看「每个 drafter 独占一卡」的 N=4/7，需要多于 3 张卡。

## dsdSim 对照（同一端云场景）

sim 用真机标定（`scripts/final_compare.sh`）：target verify/prefill 与 draft 延迟由 `sim/calibrate_from_run.py` 从真机 `steps.jsonl` 拟合（verify = 23.6 + 0.19·B + 0.079·Σ(γ+1) ms，R² 0.81；1B draft = 9.7 + 3.6·γ + 0.23·γ·(B−1) ms，R² 0.82；EAGLE3 单独拟合），接受率用真机实测的逐位置 profile（EAGLE3 用自己的），lockstep 模式（我们的 target step 阻塞等 draft；draft 作为 step 尾部），轮数按 γ/L̄ 对齐，prompt 长度用真机实际长度；端云场景的上行+下行在 lockstep 下是纯延迟，精确地给 TTFT/端到端 +RTT。EAGLE3 对照的是 sim 的链式代理（sim 自带的 EAGLE 模型在首个零接受轮就结束整个请求，不可用）。

按档位标定（N 各自的 draft 延迟，体现共卡），`real / sim (err)`：

| Method | Metric | N=1 real / sim (err) | N=2 real / sim (err) | N=4 real / sim (err) | N=7 real / sim (err) |
|---|---|---|---|---|---|
| Vanilla SD | TTFT (ms) | 138.2 / 157.8 (+14%) | 148.0 / 179.6 (+21%) | 196.7 / 223.7 (+14%) | 538.4 / 307.8 (-43%) |
|  | TPOT (ms) | 19.4 / 18.2 (-6%) | 20.3 / 19.5 (-4%) | 28.2 / 26.7 (-6%) | 65.3 / 37.7 (-42%) |
|  | Thr. (req/s) | 0.9 / 0.9 (+0%) | 1.9 / 1.9 (+0%) | 3.7 / 3.7 (+0%) | 5.9 / 6.2 (+4%) |
| SpecDec++ | TTFT (ms) | 167.5 / 189.2 (+13%) | 181.0 / 199.2 (+10%) | 293.5 / 281.0 (-4%) | 5134.9 / 360.9 (-93%) |
|  | TPOT (ms) | 19.1 / 17.6 (-8%) | 20.1 / 18.6 (-8%) | 33.8 / 28.3 (-16%) | 79.8 / 36.7 (-54%) |
|  | Thr. (req/s) | 0.9 / 0.9 (-0%) | 1.9 / 1.9 (+0%) | 3.6 / 3.7 (+0%) | 5.0 / 6.1 (+21%) |
| EAGLE3 | TTFT (ms) | 125.0 / 143.3 (+15%) | 129.0 / 158.5 (+23%) | 172.5 / 190.3 (+10%) | 363.9 / 240.3 (-34%) |
|  | TPOT (ms) | 18.4 / 16.8 (-9%) | 19.6 / 17.7 (-10%) | 26.6 / 22.2 (-16%) | 55.9 / 30.6 (-45%) |
|  | Thr. (req/s) | 0.9 / 0.9 (+0%) | 1.9 / 1.9 (-0%) | 3.7 / 3.7 (+0%) | 6.1 / 6.3 (+3%) |

MAPE 15.2%（36 格，中位数 9.6%）；draft 延迟合并标定时 18.0%（中位数 8.9%）。（接受率这里用的是逐位置概率 + 每轮独立抽样，即 dsdSim 自己的机制；下一节改为逐轮回放真机事件。）

结论：

1. **未饱和区（N ≤ 4）sim 基本准**：吞吐一致；TPOT 误差 −4…−16%；TTFT 误差 −4…+23%。
2. **低负载 TTFT sim 仍偏高 10–23%**：拆开看 decode 部分两边一致（N=1 vanilla：sim 73.3 ms vs 真机 73.0 ms），差距在 prefill 段——最可能是 prefill 成本标定：「带 prefill 的 step 额外 ≈ 25 ms」这一项的拟合 R² ≈ 0、残差 p90 ≈ 27 ms，很可能高估。
3. **饱和区（N=7）sim 严重低估**：TPOT −42…−54%，SpecDec++ TTFT −93%（真机 5.1 s vs sim 0.36 s），吞吐天花板高估最多 21%。两个缺口：(a) sim 没有 vLLM 的 `max_num_seqs` 准入上限，真机请求在 waiting 队列排队；(b) 每卡 3–4 个 drafter 进程的争用（上表）不是 draft 延迟的线性函数。**这是与仿真器结论真正不同的地方**：sim 会认为 N=7 时 SpecDec++ 仍可用，真机上它已饱和。这正是 paper 自称「组件标定、非端到端验证」留下的空白。

### 用真机 token 事件回放接受率（排除接受率建模误差）

论文主结果用的是「回放硬件采集的 `acceptance_seq`」，但公开仓库的 trace 里没有这个字段、代码里也没有回放逻辑。我们用自己真机 run 的逐轮事件（`steps.jsonl`：每请求的 γ、每轮 accepted）补上这一能力（`sim/make_config.py --replay-from`，shim 里按请求逐轮回放）：sim 的每个请求用真机的 γ、第 k 轮接受真机第 k 轮的个数、跑同样多的轮次（vanilla N=1 校验：轮数 4623 = 4623，accepted 11748 = 11748）。这样接受率完全精确，剩下的误差只来自 sim 的系统模型：

| Method | Metric | N=1 real / sim (err) | N=2 real / sim (err) | N=4 real / sim (err) | N=7 real / sim (err) |
|---|---|---|---|---|---|
| Vanilla SD | TTFT (ms) | 138.2 / 155.8 (+13%) | 148.0 / 171.0 (+16%) | 196.7 / 218.5 (+11%) | 538.4 / 297.9 (-45%) |
|  | TPOT (ms) | 19.4 / 19.2 (-1%) | 20.3 / 20.3 (+0%) | 28.2 / 27.9 (-1%) | 65.3 / 39.3 (-40%) |
|  | Thr. (req/s) | 0.9 / 0.9 (+0%) | 1.9 / 1.9 (+0%) | 3.7 / 3.7 (-0%) | 5.9 / 6.1 (+3%) |
| SpecDec++ | TTFT (ms) | 167.5 / 186.1 (+11%) | 181.0 / 198.5 (+10%) | 293.5 / 273.2 (-7%) | 5134.9 / 355.3 (-93%) |
|  | TPOT (ms) | 19.1 / 19.0 (-1%) | 20.1 / 19.5 (-3%) | 33.8 / 29.4 (-13%) | 79.8 / 38.1 (-52%) |
|  | Thr. (req/s) | 0.9 / 0.9 (-0%) | 1.9 / 1.9 (+0%) | 3.6 / 3.7 (+0%) | 5.0 / 6.1 (+23%) |
| EAGLE3 | TTFT (ms) | 125.0 / 135.1 (+8%) | 129.0 / 149.3 (+16%) | 172.5 / 175.6 (+2%) | 363.9 / 223.8 (-38%) |
|  | TPOT (ms) | 18.4 / 17.5 (-5%) | 19.6 / 18.5 (-6%) | 26.6 / 23.1 (-13%) | 55.9 / 31.6 (-44%) |
|  | Thr. (req/s) | 0.9 / 0.9 (+0%) | 1.9 / 1.9 (+0%) | 3.7 / 3.7 (+0%) | 6.1 / 6.3 (+3%) |

MAPE 13.3%（36 格，中位数 5.5%）。T1（N ≤ 2）下 TPOT −6…0%、吞吐一致；TTFT 仍偏高 8–16%（prefill 段，与接受率无关）；N=4（共卡）TPOT −13%；N=7 仍严重低估——这些就是 dsdSim 系统模型本身的误差。

### 对照过程中发现并修正的问题（均在我们的工具里，不在 dsdSim 原版）

- **lockstep 的 draft 位置**：vLLM 一个 step 是「forward（verify + prefill）→ propose（draft + RTT）」，draft 在尾部；我们的 sim 适配层最初把 draft + RTT 算在 batch 头部，新请求的 prefill 所在 batch 要白等别人的 draft、首轮还要再付一次自己的——TTFT 平白多一轮（≈ 34 ms）。改为「forward 结束即完成、target 再占用 draft 尾巴」后，低负载 TTFT 误差从 +33…59% 降到现在的水平，TPOT/吞吐不变（`sim/dsdsim_shim.py`）。
- 早先文档里「sim 由 drafter 驱动，首 token 前多一次网络往返和 draft」的解释对 lockstep 对照不成立（该模式下 prefill 网络为 0）；它描述的是 dsdSim 原生异步模式下的端云场景，现已通过真机 `--client-side` 在两边统一。

## 数据中心场景（对照）

请求直接到达 target（不经客户端上下行），其余相同，结果在 [results/table3_final/](../results/table3_final/)：TTFT 比端云场景少约一个 RTT（如 vanilla N=1：dsdSim 口径 132 vs 138 ms），TPOT/吞吐在噪声内一致；sim 对照（同样修正后）MAPE 14.4%（中位数 8.5%）。

## n:m（2 target : 2 drafter，3 张卡）

2 个 target 各占一卡，2 个 1B drafter 共用第三张卡（每个 target 都能扇出到两个 drafter），Vanilla γ=4，RTT 10 ms，router 按到达时间派发，总负载 8 req/s，400 请求：

| 路由 | 分流 t0/t1 | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | Thr. (req/s) | L̄ | 记账 |
|---|---|---|---|---|---|---|
| RR | 200 / 200 | 147.6 / 232.4 | 27.2 / 37.7 | 6.82 | 3.55 | ✅ |
| JSQ | 199 / 201 | 154.3 / 237.6 | 27.6 / 39.0 | 6.82 | 3.53 | ✅ |

这个负载低于饱和，两种策略几乎一样，符合 paper「策略差异出现在 target 饱和附近」的叙述；要复现 JSQ/RR 交叉需要按 ρ 扫到饱和（plan.md V3），不在本次范围内。

实现上踩到的坑：第一版 JSQ 在每次派发前同步问一遍各 target 的队列深度，而 target 只在两个 engine step 之间回复，等于每个请求在 router 上多等一个 step——同样负载下 TTFT 均值 434 ms、p99 1.2 s（RR 146 ms）。现在 target 每个 step 后主动推送深度，router 用「快照 + 快照之后已派出的请求」，不再阻塞（`dsdreal/router/`）。

## 复现

```bash
# remote（NYULAB）：门禁 + Table 3（默认端云场景；SCENARIO=datacenter 切回），等空闲卡、逐 run 查卡、可续跑
nohup scripts/run_all_nyulab.sh > runs/pipeline.log 2>&1 &
# 本地：拉回结果、标定 sim、出对照表，并打包进 results/
REMOTE_RUN=table3_client scripts/final_compare.sh runs/table3_client_final
python scripts/pack_results.py runs/table3_client_final results/table3_client
```
