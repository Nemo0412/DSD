# dsd-real：分布式投机解码真实系统（DSD-Sim 的端到端验证）

## Context

`22045_DSD_Sim_A_Simulation_Fra.pdf`（DSD-Sim, arXiv:2511.21669）是一个**离散事件仿真器**：SimPy 建模 N 个 draft 节点 : M 个 target 节点，VIDUR 预测 GPU 延迟，可配置 RTT/jitter/fat-tree 网络，可插拔 routing / batching / speculation-window 策略，回放硬件采集的 `acceptance_seq`。它自己的 Limitations 明写：

> DSD-Sim is component-calibrated rather than end-to-end validated against a physical multi-node speculative decoding stack, so its largest-scale metrics should be interpreted as system-level projections.

`dsd-real` 补的就是这一句。目标**不是**造一个更快的 serving 系统，而是造一个**指标口径与仿真器一一对齐的、可复现的验证台**。所有取舍据此：**可控性和可观测性 > 峰值性能**。

用户已确认：租单机多卡即可；采样设置以 paper 为准（paper 未指定 temperature，本计划按 T=0 greedy 走，见 §6）。

---

## 一、直接回答："server 用 nanovllm 改还是真 vllm"

**用真 vLLM，而且不用 fork —— 全部通过官方插件点接入。nano-vllm 不做主线，连备选都不是（备选是"fork vLLM 打 50 行 patch"）。**

### 决定性理由：论文的两个核心结论都是「真实 continuous batching 引擎在排队压力下的涌现行为」

> "JSQ … near saturation, JSQ can over-concentrate requests on temporarily short queues, causing head-of-line blocking and higher TPOT."
> "both policies eventually reach similar request-throughput ceilings … bounded by target-side compute capacity."

这两条**不是 SD 算法的性质**，是 target 侧真实调度器 + paged KV + 抢占 + 准入控制在饱和点附近的性质。

- vLLM 白给你这些，外加已经埋好的 per-request TTFT/TPOT/queue-time 和 acceptance 指标。
- nano-vllm 要你自己写 800–1200 行调度器，然后拿**你自己发明的排队行为**去"验证"仿真器。审稿人一句"JSQ 反转是分布式 SD 的性质，还是你那 400 行玩具调度器的性质？"就没法答了。**这一条就否决 nano-vllm，不用比工作量。**

nano-vllm 唯一的真优势（varlen prefill 天然支持 batch 内变长 γ、免 padding）vLLM 本来也有——ngram 路径就返回逐请求变长的 draft。

### 已核实：vLLM 0.21.0（你 pegainfer venv 里现装的）就够用，**不必升级**

我逐条 grep 过 `/home/scatyf3/pegainfer/.venv/.../vllm`：

| 事实 | 核实结果 |
|---|---|
| `NgramProposer.propose(sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None) -> list[list[int]]` | ✅ `v1/spec_decode/ngram_proposer.py:131`，签名完全一致 |
| dispatch 处只做 `assert isinstance(self.drafter, NgramProposer)` | ✅ `v1/worker/gpu_model_runner.py:4604` → **子类可通过** |
| KV 回滚是 O(1) | ✅ `v1/core/sched/scheduler.py:1325` `request.num_computed_tokens -= num_rejected`，不释放 block |
| `scheduler_cls` 是一等公民插件点 | ✅ `config/scheduler.py:127`，接受 `"mod.custom_class"` 字符串 |
| `method="custom_class"`（PR #39487, 合并于 2026-05-13） | ❌ **0.21.0 里没有**。`SpeculativeMethod` 只有 ngram/medusa/mlp_speculator/draft_model/suffix/eagle/ngram_gpu。最新 release 是 v0.27.1 (2026-08-11) |

所以有两条等价的接入路径，**先走第一条，把"必须升级到某个版本"从关键路径上拿掉**：

```python
# dsdreal/target/proposer.py
from vllm.v1.spec_decode.ngram_proposer import NgramProposer

class RemoteDraftProposer(NgramProposer):
    """继承只为过 isinstance 断言；用 ZMQ 往返替换 ngram 查表。"""
    def propose(self, sampled_token_ids, num_tokens_no_spec,
                token_ids_cpu, slot_mappings=None) -> list[list[int]]: ...
```

启动时 `--speculative-config '{"method":"ngram","num_speculative_tokens":8,"prompt_lookup_max":8}'`（`config/speculative.py:587` 会令 `draft_model_config = target_model_config`，**target 进程里不会真的加载第二个模型**），再用 `vllm.general_plugins` entry point 猴补 `GPUModelRunner.__init__` 换掉 `self.drafter`。`dsdreal/target/compat.py` 探测到 `custom_class` 可用时自动切原生路径：

```bash
python -c "import typing; from vllm.config.speculative import SpeculativeMethod as M; print('custom_class' in typing.get_args(M))"
```

### 顺带纠正 context.md 里三处判断

1. **"nanovllm 只有一次发到底的 generate"——错。** `LLMEngine` 有 `add_request()` + `step()`，`generate()` 只是 `while not is_finished(): step()` 的包装。（但结论不变，仍不用它。）
2. **"kv 回滚管理"这个工作量不存在。** paged KV 里回滚 = 逻辑长度减法，物理 block 不动，被拒 KV 下一步被新 `slot_mapping` 覆盖。
3. **"padding+mask 还是 varlen 打包"不用纠结。** vLLM 的 spec 路径本来就逐请求变长。
4. **RFC #42109 是空的。**（@laviier，2026-05-08 开，零评论零 PR）它提的架构和你 context.md 26–32 行推出来的是同一个东西——你的判断对，但没人替你写。也没有可 fork 的先例：DSI 自己是仿真器，DSD-Sim 未放码，FlowSpec 是 pipeline 并行，SGLang 的 `STANDALONE` 仍是同进程。

---

## 三、架构

控制循环放在 **target 引擎**（不是字面照搬论文 Fig.1(b) 的 draft-driven 叙事）：

```
              Router frontend：routing policy (Random | RR | JSQ) 在准入时选 target
                        │                             │
              ┌─────────▼─────────┐        ┌──────────▼────────┐
              │ Target 0 (vLLM)   │        │ Target 1 (vLLM)   │   M targets
              │ LABScheduler      │        │ LABScheduler      │
              │ RemoteDraftProposer│       │ RemoteDraftProposer│
              └────┬────┬─────────┘        └────┬────┬─────────┘
          ZMQ DEALER │  └── NetworkEmulator ────┘    │
              ┌──────▼──┐ ┌────────┐ ┌────────┐ ┌───▼────┐
              │Drafter 0│ │Drafter1│ │Drafter2│ │Drafter3│   N drafters (4:1)
              │ ROUTER  │ │        │ │        │ │        │
              └─────────┘ └────────┘ └────────┘ └────────┘
```

每个 engine step，`RemoteDraftProposer.propose()`：
1. 收到 `sampled_token_ids` —— **这就是 accept/reject 反馈**（rejection sampler 的解析输出，`len(sampled_token_ids[i]) == num_accepted_i + 1`），和上一步提议的 diff 就是 acceptance bits。**不需要另开反馈通道。**
2. 按 `drafter_id` 分组，**并发** DEALER 发送 `{seq_id, accepted_tokens, gamma_i}`
3. `poll()` 等齐 → 墙钟 = `max_i(RTT_i)` + max drafter 计算，不是求和
4. 返回逐请求变长的 `list[list[int]]`

`RemoteDraftProposer.propose()`是插件搞得吗

### 能从引擎里抓到什么：三层，别混

**vLLM 里没有一次单独的 "verify forward"。** 每个 engine step 就是一次 forward，对每请求覆盖 (γ+1) 个位置，同时完成「验证 γ 个 draft token」和「产出下一步 logits」。所以"verify 结果"就是这一步 sampler 的输出。

| 层 | 抓什么 | 怎么抓 | 稳定性 |
|---|---|---|---|
| **L1** | `sampled_token_ids`：每请求本步产出的 token，`len-1 == num_accepted` | `propose()` 入参，白给 | ✅ 官方契约，有测试覆盖 |
| **L2** | target 在**全部 γ+1 个位置**的原始 logits + `SpecDecodeMetadata`（`draft_token_ids`, `num_draft_tokens`, `cu_num_draft_tokens`, `bonus_logits_indices`, `target_logits_indices`） | 在 `register()` 里包 `RejectionSampler.forward`（`v1/sample/rejection_sampler.py:87`，调用点 `gpu_model_runner.py:3419`） | ⚠️ 内部实现，版本敏感 |
| **L3** | verify forward 自身耗时（与 `propose()` 里的 ZMQ 往返分离） | CUDA event 包 `execute_model` —— GPU 异步，`perf_counter()` 量不准 | ⚠️ 内部实现 |

L1 够算 acceptance ratio、mean accepted length、γ 决策——**AWC 需要的反馈信号全在 L1。**

**但 L2 不是可选的：** §5 定的 `acceptance_seq` 用 `full_window` 约定，首次拒绝之后那些位的 bit 是"target 在该位置的 argmax 是否等于 draft token"，`sampled_token_ids` 里没有，**只能从 `logits` 算**。若作者确认 DSD-Sim 吃 `prefix` 约定，L1 就够，L2 退化为只服务 V5。→ **这就是 §11 第 1 问必须在 Stage 0 问掉的原因。**

L2/L3 碰内部实现，**静默失效比崩掉危险**（会让 `acceptance_seq` 悄悄少一半位）。故 Stage 0 增加 **K7**：断言包好的 hook 确实被触发，且捕获到的 `metadata.num_draft_tokens` 之和等于本步 γ 之和。

**为什么这和论文的图等价：** request 级粒度下事件序列完全相同（draft 计算 → 网络 → verify 计算 → 网络 → 回滚），每段都有自己的时间戳对。`while` 循环的物理位置对 request 级仿真器不可见。**在方法论章节明写这一点**，是可辩护的。

**好处：** routing（请求→target，准入时决策）和论文 1:1 对应；JSQ 读的队列深度是 vLLM 的真数（`Scheduler.get_request_counts()`），不是虚构的。

**fused 模式** = 同一套代码换 `LocalDrafter`（进程内、无 ZMQ、无 emulator），配置里一个 `execution_mode: fused | distributed` 切换。

### LAB batching：vLLM 会跟你抢 batch 组成——答案是在**准入端**做，不是在 `self.running` 上做

`Scheduler.schedule()`（`scheduler.py:310`）里 batch 组成由 (a) `self.waiting` 的出队顺序、(b) token budget 压力下 `self.running` 的顺序 决定。

**在 (a) 做。** 两个理由：
1. **对齐**：DSD-Sim 是 request 级的，没有常驻 GPU batch 可重排，它的 LAB 必然是准入时分组。重排 `self.running` 会实现一个仿真器根本不建模的策略——那是反目标的。
2. **安全**：`self.running` 顺序牵连 `InputBatch` 持久 slot、condense、抢占（`scheduler.py:462` 抢占的是**最后一个**）。爆炸半径大、对齐价值低。

```python
# dsdreal/target/scheduler_lab.py
class LABRequestQueue(RequestQueue):     # v1/core/sched/request_queue.py 的 ABC
    """队首请求定义长度桶；batching window 内优先弹同桶请求。"""
class LABScheduler(Scheduler):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        if dsd_env.batching_policy == "lab":
            self.waiting = LABRequestQueue(...)
```
启动：`--scheduler-cls dsdreal.target.scheduler_lab.LABScheduler`，零 fork。

长度估计器要诚实：回放时用 trace 里的 `prompt_length + output_length`（oracle LAB，和仿真器持平），同时提供 `--lab-oracle false` 的在线估计版，**两个都报，差值本身是个保真度数字**。

---

## 四、仓库结构

```
dsd-real/
├── pyproject.toml                    # entry-points."vllm.general_plugins"
├── constraints/vllm-0.21.0.txt       # 精确 pin
├── configs/{local_fused_1x1,local_dist_1x1,rented_2t_8d,sim_parity_fattree_4096}.yaml
├── configs/schema/dsd_config.schema.json    # 与 DSD-Sim 的契约
├── data/legacy_calibration/          # spec/output/*.json 冻结副本
├── data/traces/*.jsonl
├── dsdreal/
│   ├── config/{model,auto_topology,loader,to_sim_yaml}.py
│   ├── proto/{messages,transport}.py       # msgspec.Struct + DEALER/ROUTER
│   ├── netem/{base,sleep,tc,null}.py
│   ├── draft/{server,engine_vllm,engine_hf,engine_replay}.py
│   ├── target/{plugin,proposer,scheduler_lab,compat,launcher}.py
│   ├── policy/{base,routing,batching,window}.py
│   ├── router/{frontend,registry}.py
│   ├── workload/{trace,build_traces,arrival}.py
│   ├── metrics/{records,collector,gpu_probe,export,compare}.py
│   └── experiments/driver.py + sweeps/*.yaml
├── scripts/{verify_vllm_hooks,preflight,run_local_smoke,run_rented_batch}.{py,sh}
└── tests/{test_lossless,test_kv_rollback,test_variable_gamma,
           test_metric_parity,test_acceptance_seq_roundtrip,test_auto_topology}.py
```

策略接口按论文原文（*"read-only snapshot of … queue depth, RTT, TPOT, and acceptance rate"*）字面实现：

```python
@dataclass(frozen=True, slots=True)
class SystemSnapshot:
    t_ms: float
    queue_depth: Mapping[str, int]            # target_id -> waiting+running
    rtt_ms: Mapping[tuple[str, str], float]   # EWMA
    tpot_ms: Mapping[str, float]
    acceptance_rate: Mapping[str, float]
    accepted_len: Mapping[str, float]

class RoutingPolicy(Protocol):  def select_target(self, req, snap) -> str: ...
class BatchingPolicy(Protocol): def order_key(self, req, head, snap) -> tuple: ...
class WindowPolicy(Protocol):   def choose(self, req, snap) -> tuple[int, ExecMode]: ...
```
`WindowPolicy.choose -> (gamma, exec_mode)` 与论文逐字对应（*"selects the speculation window size (γ) and execution mode (fused vs. distributed)"*）。实现：`StaticGamma(4)`、`DynamicGamma`、`AWC`、`SpecDecPP(gammas=(2,4,6,8))`。

---

## 五、指标对齐（最大的风险点，代码之前先定死）

| 指标 | dsd-real 定义 | 对齐陷阱 |
|---|---|---|
| **TTFT** | `first_token_ts − arrival_time`，`arrival_time` 在 **router frontend 入口**打戳 | vLLM 自己的 `RequestStateStats.arrival_time`（`v1/metrics/stats.py:202`）从 engine-core 准入起算，**会漏掉 routing + 排队等待**。必须覆盖。**头号陷阱。** |
| **TPOT** | `(last_token_ts − first_token_ts) / (num_output_tokens − 1)`，按**接受的 token**算，不是按 engine step | `spec/metrics.py:126` 已经修对了，别改回去 |
| **acceptance ratio** | `Σ num_accepted / Σ gamma`，**不含 bonus token** | vLLM 的 `mean_acceptance_length = 1 + accepted/drafts` **含** bonus。`spec/README.md` 记录了为这个坑烧掉一整天。**两个都导出并标注** |
| **request throughput** | `num_finished / 稳态窗口`（掐头去尾各 10%） | 论文报 req/s，`spec` 只报过 tok/s |
| **target utilization** | NVML `utilization.gpu` 100 Hz 采样 + vLLM `SchedulerStats.kv_cache_usage` **两个都导** | 仿真器的 "target device utilization" 语义有歧义，两个都有才能对上 |
| **network queueing delay** | `Σ (recv_ts − send_ts − configured_rtt_ms)`，即**超出注入 RTT 的部分** | 这就是 `spec` 里一直是 `null` 的那个。定义成"超出量"才能和仿真器的链路队列延迟直接比 |

`dsd_metrics.json` 顶层：`{schema_version, run_id, config_digest, config, git, env, steady_window_ms, system_metrics{...}, per_request[{request_id, drafter_id, target_id, exec_mode, routing_decision{policy, chosen, queue_depth_at_decision}, arrival_time_ms, prompt_length, output_length, ttft_ms, tpot_ms, e2e_latency_ms, acceptance_ratio, mean_accepted_len, acceptance_seq, acceptance_seq_convention, spec_steps[{i, gamma, num_accepted, t_draft_start/end_ms, t_send/recv_ms, t_verify_start/end_ms, batch_size_at_verify, net_excess_ms}], gamma_decisions}]}`。`spec_steps` 是一切的原料；量大时落 `steps.jsonl`，用 `--emit-steps {full,sampled,none}` 控制。

### `acceptance_seq` 有个必须先问清的歧义

论文 Table 1 给的是一条扁平 bit 向量，*"replayed token-by-token during verification"*。两种约定：
- **(A) `full_window`**：每步记全部 γ 位（含首次拒绝之后的位），长度 `Σ γ_i`
- **(B) `prefix`**：只记到首个 `0`，长度 `Σ (num_accepted_i + 1)`

token-match 接受下 target 本来就在 γ+1 个位置都算了 argmax，所以 (A) 能产出。**按 (A) 导出并显式写 `"acceptance_seq_convention": "full_window"`，同时导出 `spec_steps[].{gamma, num_accepted}`**——(A)+γ 可无损转 (B)，反过来不行。提供 `trace.to_prefix_convention()`。然后**发一封三行邮件问原作者 DSD-Sim 吃哪种**，五分钟的问题，猜错要重跑。

**更深的问题也要说破：** 回放 `acceptance_seq` 预设了接受行为只是 `(draft_model, target_model, prompt)` 的函数。它不是——改 γ 会改变拒绝点，文本发散，bit 就失效了。论文在 SpecDec++ 的 γ∈{2,4,6,8} 上照样回放。**真机能量化这个误差，别人不能**——见下面的 V5，这是本项目最新颖的一块，而且几乎免费。

---

## 六、网络仿真：默认必须是 sleep，不是 tc netem

关键认识：**DSD-Sim 自己的网络模型就是个 sleep**（*"Network links are represented as delay elements … parameterized by configurable RTT and jitter"*），NS-3 是用来**标定**这个抽象的，不是替代它。

所以要验证 DSD-Sim，主路径必须**复现它的抽象**。默认用 `tc netem` 反而会引入 TCP 拥塞控制、真实排队、带宽限制——仿真器根本不建模的东西，测出来的偏差说明不了它对自己模型的保真度。

- **默认 `SleepDelayEmulator`（对齐模式）**：`proto/transport.py` 在 send/recv 两侧各 `await asyncio.sleep(base_ms/2 + jitter)`。抖动分布要和仿真器一致（多半是截断正态，配置项 `jitter_dist: {normal, uniform, none}`，**问作者确认**）。`asyncio.sleep` 有 ~50–200 µs 粒度，RTT ≥ 0.5 ms 时 <5% 误差；更小的用 `sleep(d-200µs)` + 自旋。**把 emulator 自己的"实际 vs 请求"延迟分布测出来放附录**，先手堵掉审稿人必问的一句。
- **`TcNetemEmulator`（真实性消融）**：需 `NET_ADMIN`；单机多进程时 `lo` 上 netem 不直接生效，要每个 drafter 一个 netns + veth 对。跑**一个**配置对比，报 `Δ(TPOT_netem, TPOT_sleep)`——这个数把论文的 NS-3 标定延伸到真硬件，是实打实的增量贡献。

`config/auto_topology.py` 实现论文的 fat-tree（pods/racks/nodes + `same_rack_us`/`cross_rack_us`/`cross_pod_us`）→ `LatencyMatrix[node][node]`。每个进程在虚拟 fat-tree 里有个 `node_id`，**与物理机器无关**——所以 `rented_2t_8d.yaml` 可以在 2 台物理机上把 8 个 drafter 摆进 3 个虚拟 pod。`tests/test_auto_topology.py` 断言矩阵能复现 Table 2 的节点数（64…4096）和 concentrated/distributed 两种摆放。另提供 `network.mode: matrix` 直接加载节点对延迟矩阵，作为复现某个仿真配置的逃生口。

---

## 七、规模差距：把 N 换成 ρ

论文 Fig.5–8 扫 400–2000 个 draft 节点、100 个 target worker。你只有个位数。但注意论文自己的措辞：*"scale 400–2,000 **logical** single-GPU draft workers"*——**它本来就是逻辑抽象。**

更要紧的是：**N 不是真正起作用的自变量。** 两个头条结论的物理都是 **ρ = 供给负载 / target 容量**："near target saturation"、"bounded by target-side compute capacity"。所以：

**把 x 轴换成 ρ（用实测的 target 饱和吞吐归一化），N 降为次要因子。** ρ ∈ [0.2, 1.4] 用 **2 个 target + 4–8 个 drafter** 就能全覆盖——饱和 target 靠加 λ，不靠加 drafter。这样对比是尺度无关的，仿真器可以在你的 N 和它的 N 上都跑一遍。

（顺带：论文 Fig.5 里吞吐随 N **上升**，说明那 50 req/s 要么是 per-drafter 要么供给随 N 增长，文本有歧义——一并问作者。）

**三档 draft worker，每张图都标明档位：**

| 档 | 是什么 | 可用于 | **不可**用于 |
|---|---|---|---|
| **T1 物理** | 1 drafter = 1 独占 GPU | 全部 | N > ~6 |
| **T2 逻辑复用** | K 个独立进程共享一卡（MPS），各自 KV 池/队列/虚拟 node_id | 排队、路由、批处理、网络 | 任何关于 draft 侧算力竞争或 draft GPU 利用率的结论 |
| **T3 回放** | `engine_replay.py`，无 GPU，回放 trace + 按 T1 拟合的 `f(γ, batch, seqlen)` sleep | 网络拓扑、路由、百级 N 的排队 | 接受行为、模型质量、draft 计算 |

**可辩护性铁律：每张 T2/T3 图必须配一次重叠验证**——N ∈ {1,2,4} 在 T1/T2/T3 上同配置跑，TPOT/吞吐在声明的容差内一致。做成自动报告 `metrics/compare.py --tier-overlap`。审稿人接受复用的前提就是你出示了这个检查。T2 另加护栏：draft GPU 利用率 > 60% 时导出直接标 `"tier_validity": "INVALID"`。

---

## 八、验证图表集（这才是交付物）

**V1 GPU 标定**（对应 Fig.4）：导出 VIDUR schema 的 `latency_profile.json`，叠 VIDUR 预测。**用 `spec/output/` 的现成 A40/A100/H100 数据，成本≈0。**

**V2 接受率敏感性**（对应 Table 4）★ **单位成本价值最高，先做。** 加 `ForcedAcceptanceDrafter` 按比例污染 draft token 命中目标 L̄，扫论文那五个点（L̄ 1.20→2.72，TPOT 30.5→18.6 ms，tok/s 1015→1285）。**单卡单 target 的 1-D 扫描，不涉及路由和 N:M。** 它直接检验仿真器的核心耦合（接受率→轮数→验证负载→批处理→TPOT/吞吐）。曲线形状对上且误差 ≤15% 才谈下游。**在本地 5070 Ti 上跑，租卡之前。**

**V3 路由**（对应 Fig.5–6）：**需要 M ≥ 2**（M=1 时路由是空操作——这是租卡规格的硬下限）。{Random, RR, JSQ} × ρ{0.3…1.3} × {gsm8k, cnndm, humaneval}。**成功判据不是绝对误差，是定性一致**：① 各 ρ 下策略排序一致；② 真机上**存在** JSQ/RR 交叉点且 ρ\* 落在声明区间内；③ 绝对 TPOT 误差 ≤20%。②是论文的头条主张——**在真硬件上（哪怕 N=8）确认交叉存在就是强结果；确认不存在是更强、更有意思的结果**，要准备好照实报。

**V4 批处理**（对应 Fig.7–8）：只需 M=1，最便宜。FIFO vs LAB，验两条：LAB 降 TPOT、两者吞吐天花板相同。**LAB 增益要同时报绝对 ms 和 TPOT 占比**（论文的"1–2 ms"是在 20–28 ms 上，≈5–8%；你的硬件 TPOT 不同，可比的是比例）。oracle/online 估计器两个变体都跑。

**V5 `acceptance_seq` 的 γ-不变性**（无仿真器对应项，**全新**）：在 γ=4 录 trace，然后在 γ∈{2,6,8} 和 AWC 下实跑，测实际接受 bit 与 γ=4 录制的偏离（bit 流编辑距离、ΔL̄、ΔTPOT），再让 DSD-Sim 回放 γ=4 trace 跑同样的 γ 做对比。**直接量化回放方法论的保真度代价。** 便宜、新颖，把论文的一条 limitation 变成一个测出来的数。

**V6 头条误差表**：论文格式的 8–12 行 `Config | Metric | Real | DSD-Sim | Rel.err`，加一张 sim-vs-real 散点 + y=x + MAPE。**别承诺 3%**——论文的 3.2–4.4% 是**组件**误差；端到端 10–25% 是正常且诚实的结果。贡献的表述是："首次对 DSD-Sim 做端到端验证，端到端误差 X%，四条策略结论均在趋势层面吻合。"

**V7 外推边界图**：一条 N=8 的竖线，左边实测、右边仿真投影加阴影，图注写明外推假设。这是**正面回应**论文的 limitation，而不是假装消灭了它。

**租卡规格**（由 V3 的 M≥2 倒推）：**单机 4× A100-40G 或 4–8× L40S/A6000**。单机多卡**优于**多机——网络本来就是仿真的，单机反而消除了控制不了的真实网络方差。多机只留一次确认性对比（单机 sleep 仿真 ≈ 真实跨节点 RTT）。预算：真做到无人值守的话 ~10–16 GPU-hour，A100 spot 约 $50–130。**主要风险不是算力钱，是在租来的时间上 debug。**

---

## 九、正确性验证

**losslessness（`tests/test_lossless.py`）**：`spec/README.md` 已记录这个坑——*"使用bf16可能导致accept rate != 100%"*。**根因不是 SD bug**：bf16 下 logits 依赖归约顺序，归约顺序依赖 batch 组成，baseline 跑和 SD 跑的 batch 组成不同，近似平局处 argmax 翻转。vLLM 的非投机 greedy 本身也不是 batch-invariant 的。

所以：`temperature=0, --enforce-eager, VLLM_ENABLE_V1_MULTIPROCESSING=0`，固定种子，200 条 GSM8K × γ∈{1,2,4,8}。**fp32 断言逐 token 相同（这是无损性证明）；bf16 只断言首次分歧位置分布和 ≥99% 前缀一致率并记录**。绝不写一个断言 bf16 精确相等的测试——它会永远 flake，而且你已经知道原因了。

**范围声明**：`draft_probs=None` → token-match 接受，**T=0 无损，但不是 Leviathan 随机拒绝采样**。明写：**dsd-real 验证的是 greedy/T=0 的分布式 SD**；T>0 需要把 draft 概率也传过网络并 patch model runner，v1 范围外，列为 limitation。

**KV 回滚（`test_kv_rollback.py`）**：① `PoisonDrafter` 在位置 k 故意给错 token，断言输出仍与 baseline 逐 token 相同且 `num_accepted == k`；② 连续 20 步强制位置 0 拒绝，断言 KV block 用量不超过非投机 baseline（O(1) 回滚证明）；③ **prefix cache 安全**：开 `enable_prefix_caching`，先跑重度拒绝的请求 A，再跑同 prompt 的请求 B，断言 B ≡ A ≡ baseline（"未确认投机 token 被封进前缀缓存"这个隐患）。

**变长 γ（`test_variable_gamma.py`）**：一个 batch 里 8 个请求 γ=[1..8]，断言各自输出与单独同 γ 跑一致；再断言 JSON 里 `spec_steps[i].gamma` 是**实际用的** γ 而非配置的上限——这条抓的是"vLLM 把所有 draft padding 到 `num_speculative_tokens`"的静默失败。

**指标自洽（`test_metric_parity.py`）**：移植 `spec/test/test_metrics_validation.py` 的五条 + `Σ num_accepted + num_steps == output_length`（bonus 记账）+ `len(acceptance_seq) == Σ gamma`（full_window）+ 由 `per_request` 时间戳重算 `request_throughput_rps` 与内联值一致。

---

## 十、分阶段、工作量、终止判据

### Stage 0 — Hook 验证（0.5 天）**最先做，一行 `dsdreal/` 代码都别写**

`scripts/verify_vllm_hooks.py` 六个探针，每个都是终止判据：

| # | 探针 | 触发即终止 |
|---|---|---|
| K1 | `FakeProposer` 返回**变长**列表 `[[1],[1,2,3],[],[1,2]]` | vLLM 若断言 `len == num_speculative_tokens` → 不 fork 就做不了 AWC/SpecDec++ |
| K2 | `SleepProposer` 里 `time.sleep(0.005)` | TPOT 增量必须 ≈5 ms。否则 `propose()` 不在关键路径上，整个延迟模型作废 |
| K3 | `NoOpScheduler(Scheduler)` 经 `--scheduler-cls` 加载 | 管道不通 → LAB 需要 fork |
| K4 | `LABRequestQueue` 替换 `self.waiting`，跑 100 请求 | 输出必须仍正确；`RequestQueue` 契约若比 ABC 更严，退到 `policy="priority"` |
| K5 | `vllm.general_plugins` entry point 确实在 worker 进程里加载 | 退到 `VLLM_PLUGINS` + `sitecustomize.py` |
| K6 | `typing.get_args(SpeculativeMethod)` 在 0.21 vs ≥0.22 | 决定 `compat.py` 走子类 shim 还是原生 `custom_class` |
| K7 | 包住的 `RejectionSampler.forward`（L2）确实被触发，且捕获的 `Σ metadata.num_draft_tokens` == 本步 `Σ γ` | L2 静默失效 → `acceptance_seq` 悄悄少位。若拿不到 L2，`full_window` 约定做不了，只能退 `prefix` 约定并和作者确认可接受 |

**K1 或 K2 失败 → vLLM 插件路径在 γ 控制上死了。** 回退顺序：**① fork vLLM 到固定 tag，patch 两处 dispatch（`gpu_model_runner.py:534` 和 `:4600`），维护 ~50 行 patch series——这是正确的 pivot，不是 nano-vllm。** ② 只有 fork 都维护不下去才考虑 nano-vllm，且那时必须把结论降级为"在研究原型调度器上的趋势级验证"。

### Stage 1 — 骨架、schema、配置（2–3 天，不用 GPU）
`config/`、`proto/messages.py`、`metrics/records.py`、`workload/trace.py`、`auto_topology.py`、JSON Schema。**冻结 `spec/output/*.json`。schema 先于代码**——别的都在"指标能对上"的下游。

### Stage 2 — fused 单卡 + 无损性（3–4 天）
`target/plugin.py`、`target/proposer.py`、`LocalDrafter`、`metrics/collector.py`。**门禁：fp32 在 γ∈{1,2,4,8} 逐 token 相同。**

### Stage 3 — 分布式 1:1 + 网络仿真（4–5 天）
`draft/server.py`、`proto/transport.py`、`netem/sleep.py`。RTT 扫描（0/0.25/0.5/1/2/5 ms）**TPOT 必须随 RTT 线性上升，斜率 ≈ 1/L̄**——不线性说明 `propose()` 没在关键路径上（K2 的运行时复查）。**门禁：`run_local_smoke.sh` 15 分钟内绿；`aggregate_network_queueing_delay_ms` 非 null 且合理。**

### Stage 4 — 策略、N:M、本地跑 V2（1–1.5 周）
`router/`、`policy/*`、`scheduler_lab.py`、`engine_replay.py`、`ForcedAcceptanceDrafter`。**在 5070 Ti 上跑完 V2 再租卡。门禁/go-no-go：V2 的真实曲线若在任何绝对误差下都复现不出仿真器的单调形状，停下来查——租 GPU 去放大一个坏掉的核心关系是最贵的错误。** 同时跑 N∈{1,2} 的 T1/T2/T3 重叠检查。

### Stage 5 — 租卡批处理（1–2 天准备，~10–16 GPU-hour）
**铁律：租来的时间是批处理时间，零交互式调试。** 结构上强制：`run_rented_batch.sh` 吃 sweep manifest 无人值守跑完；先跑 `preflight.sh`（模型下载、在租来的硬件上重跑 K1–K6、2 分钟冒烟、`nvidia-smi topo -m`、`tc` 能力检查）——**preflight 失败就在花掉 10 分钟而不是 4 小时的时候退出**；每个配置独立可续跑；raw jsonl 增量 rsync 出来；开跑前先打印总时长估算。

Manifest ~40–60 配置，每个 3–6 分钟：V3 63 个（M=2, N=8 T2）／V4 30 个（M=1）／V5 5 个／V6 12 行／tier-overlap 9 个／netem-vs-sleep 4 个。**要砍先砍 V3（最贵），V2 和 V4 最便宜且价值最高。**

### Stage 6 — 仿真器对跑、对比、出图（3–4 天）
**你没有 DSD-Sim 的代码。** 要么 (a) 找作者要——**Stage 0 就发信，这是最长的杆**；要么 (b) 只对published 的 Table 2/3/4 和 Fig.5–8，那就只剩 V1、V2、V6-partial、V7。**按 (b) 规划，(a) 当惊喜。** 若拿到 (a)，`config/to_sim_yaml.py` 要早写并用 `sim_parity_fattree_4096.yaml` 验证。

**合计：兼职 ~5–6 周，GPU 租金 ~$50–150。**

### 终止 / 转向汇总

| 触发 | 决定 |
|---|---|
| K1 或 K2 在 0.21 **和** ≥0.22 上都失败 | fork vLLM 固定 tag，~50 行 patch。**不是** nano-vllm |
| K3+K4 都失败 | LAB 改走 `policy="priority"` + 长度桶优先级。弱一些但仍对齐，写进文档 |
| V2（本地，Stage 4）测不出 TPOT-vs-L̄ 的单调关系 | **停。租卡前查清。** 多半是指标 bug 不是系统 bug |
| V3 在任何可达 ρ 上都没有 JSQ/RR 交叉 | **照实报。这是发现不是失败。** 再让仿真器在你的 N 上跑一遍——若它有交叉而真机没有，那是整个项目最有价值的结果 |
| 租来的机器 preflight 失败 | 立刻销毁实例。成本 ~$0.30 |
| 发现自己在 SSH 进租来的机器 debug | 停，抓日志，销毁，回本地修 |
| vLLM fork patch 超过 ~300 行或版本升级就崩 | 永久冻结 vLLM 版本，不追上游。commit 记进 `dsd_metrics.json.env` |

---

## 十一、Stage 1 之前要问清的六件事（都便宜，猜错都贵）

1. `acceptance_seq` 是 `full_window` 还是 `prefix`？（§5）
2. Fig.5–8 的 "50 requests/s" 是全局还是 per draft node？（图只有在供给随 N 增长时才说得通）
3. 抖动分布：normal / uniform / none？按方向注入还是按 RTT 注入？
4. "target device utilization" 是算力利用率、KV 占用率、还是忙时占比？三个都说得通且差很多
5. DSD-Sim 源码能否为验证目的拿到（NDA/embargo 均可）？**现在就问**
6. LAB 的长度估计器：仿真器用的是 trace 里的 oracle `output_length`，还是在线估计？（决定你哪个变体才是对齐运行）

---

## 十二、验证方式

```bash
# 0. 六个 hook 探针（最先，~5 min）
python scripts/verify_vllm_hooks.py

# 1. fused 单进程 + 无损性
python -m dsdreal.experiments.driver --config configs/local_fused_1x1.yaml
pytest tests/test_lossless.py tests/test_kv_rollback.py tests/test_variable_gamma.py

# 2. 分布式 RTT=0（真 ZMQ 无延迟）——隔离出 IPC 开销
python -m dsdreal.experiments.driver --config configs/local_dist_1x1.yaml \
       --override network.emulator=null

# 3. RTT 扫描——验证 emulator 本身；TPOT 必须随 RTT 线性
for rtt in 0 0.25 0.5 1 2 5; do
  python -m dsdreal.experiments.driver --config configs/local_dist_1x1.yaml \
         --override network.rtt_ms=$rtt --out runs/rtt_$rtt
done
python -m dsdreal.metrics.compare --sweep runs/rtt_* --expect linear-tpot-vs-rtt

# 4. 指标自洽 + trace 往返
pytest tests/test_metric_parity.py tests/test_acceptance_seq_roundtrip.py

# 5. 全链路冒烟（回归门禁，必须 <15 min）
bash scripts/run_local_smoke.sh
```

**本地模型对**：target = **Qwen3-1.7B**、draft = **Qwen3-0.6B**（两者 `vocab_size` 均为 151936，已核实）。注意 Xwayland 已占 ~3.4 GB / 16.3 GB，可用约 12.5 GB：`--gpu-memory-utilization` 目标 0.45 + draft 0.18 ≈ 10.2 GB。**接受率一般没关系**——这个阶段在调管道，而 V2 的 `ForcedAcceptanceDrafter` 本来就让你把 L̄ 设成任意值。Qwen3-4B 当 target 需要先关掉桌面会话，留 `configs/local_dist_1x1_4b.yaml` 但别依赖它。

**flash-attn 在 vLLM 路径上不是问题**（已核实：`from vllm.vllm_flash_attn import flash_attn_varlen_func` 可用，`torch.cuda.get_arch_list()` 含 `sm_120`，另有 flashinfer 0.6.8）。**这也是 nano-vllm 路径的一个隐藏硬成本**：`vllm.vllm_flash_attn` 只导出 `flash_attn_varlen_func`，nano-vllm 的 decode 路径还要 `flash_attn_with_kvcache`；为 sm_120 从源码编译 flash-attn 是数小时、30+ GB 内存的活。
