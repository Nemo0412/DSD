# Stage 0 — Hook 验证结果

**结论：七个探针全绿。「插 vLLM，不 fork」的计划成立，进入 Stage 1。**

环境：vLLM 0.21.0 / torch 2.11.0+cu130 / RTX 5070 Ti (sm_120, 16 GB) / Qwen3-0.6B。
全套 ~2 分钟 GPU 时间。原始数据见 `runs/stage0_probes.json`，日志 `runs/stage0.log`。

| 探针 | 结果 | 实测 |
|---|---|---|
| K1 变长 draft | PASS | 33 次调用，batch 内 γ ∈ {0,1,2,3} 并存，输出与非投机 baseline **逐 token 相同** |
| K2 propose() 在关键路径上 | PASS | TPOT 19.3 → 26.9 ms（注入 5 ms），Δ=7.6 ms |
| K3 `--scheduler-cls` | PASS | `CountingScheduler.schedule()` 被调 32 次，输出正常 |
| K4 自定义 RequestQueue | PASS | 长度分桶队列弹出 40 个请求，输出与 FCFS baseline 完全一致 |
| K5 entry point 插件 | PASS | `register()` 在 **2 个进程**（process0 + engine core）中触发 |
| K6 `custom_class` 可用性 | PASS（信息） | 0.21.0 **没有**；走 `NgramProposer` 子类 shim |
| K7 L2 tap（verify forward） | PASS | 23 次 tap，`Σ num_draft_tokens`=276 vs 提议 288，差值 = γ×4（末步提议无人消费）；**`draft_probs is None` 23/23** |

K2 的 Δ 略大于注入值（131–152%），是 `time.sleep` 粒度加测量噪声；Stage 3 的 RTT 扫描会做精确的线性度检查。

---

## 探针跑出来的四个真实约束

### 1. `NgramProposer.__init__` 会调用 `self.propose()`

`ngram_proposer.py:57` 在构造函数里跑一次 numba JIT 预热：

```python
self.propose([[]] * 1024, np.zeros(1024, np.int32),
             np.zeros((1024, self.max_model_len), np.int32))
```

**对 `RemoteDraftProposer` 的影响（两条都必须遵守）：**

- 子类的所有状态必须在 `super().__init__(cfg)` **之前**赋值，否则预热调用会撞
  `AttributeError`（K2 第一次跑就是这么挂的）。
- `propose()` 必须能在 **transport 尚未连接** 时安全返回。预热传进来的是 1024 个空
  `sampled_token_ids`，绝不能触发 ZMQ 往返。按「全空即返回全空」处理即可。

### 2. `rejection_sampler` 有第二个调用点，会污染 L2

`gpu_model_runner.py:5793` 的 `_dummy_sampler_run` 在显存 profiling 阶段用
**随机 logits + dummy metadata** 调用 rejection sampler。未过滤时 K7 数到
`Σ num_draft_tokens`=788，而真实提议只有 288。

**这正是计划里 K7 想抓的静默失效**，而且第一次跑就出现了。L2 tap 必须有
`live` 门闩（引擎构造完成后才置位），否则 `acceptance_seq` 会混进 profiling 期间的
垃圾位——不报错，只是数据是错的。

### 3. `draft_probs` 恒为 `None`

23/23 次 tap 均为 `None`，`logits` 形状 `[Σγ + batch, 151936]`。
确认计划里的范围声明：**token-match 接受，T=0 无损，不是 Leviathan 随机拒绝采样**。

### 4. 环境：两个 WSL 特有的坑

- **flashinfer JIT 直接 `subprocess.run(["ninja", ...])`**，而 venv 的 `bin/` 不在 PATH 上
  → `PermissionError: [Errno 13] Permission denied: 'ninja'`，表现为
  「Engine core initialization failed」。修法在 `scripts/env.sh`：把
  `$DSD_VENV/bin` 放进 PATH。**任何 dsd-real 的入口都要先 source 它。**
- **WSL 强制 `spawn`**（NVML 与 fork 不兼容），所以每个会构造 `LLM` 的脚本都必须有
  `if __name__ == "__main__":` 守卫，否则 spawn 重新 import `__main__` 会递归建引擎。

---

## 已落地的文件

```
scripts/env.sh                  # PATH / venv，任何入口先 source
scripts/verify_vllm_hooks.py    # K1–K7，--probe kN 可单跑
scripts/dsd_probe_sched.py      # CountingScheduler / BucketQueueScheduler / LABRequestQueue 原型
dsdreal/target/plugin.py        # entry point：register() / register_probe()
pyproject.toml                  # vllm.general_plugins 注册
runs/stage0_probes.json         # 结构化结果
```

`LABRequestQueue`（K4 里已验证可用）是 Stage 4 LAB batching 的直接原型：
队首请求定长度桶，桶内优先弹出，桶空则重新锚定队首——永不饿死。

## 下一步

Stage 1（骨架、schema、配置，2–3 天，不用 GPU）。**schema 先于代码。**
并行动作：给 DSD-Sim 作者发信问 §11 的六个问题，Q1（`acceptance_seq` 约定）决定 L2 是不是必需。
