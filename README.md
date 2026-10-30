# DSpec

DSpec studies disaggregated speculative decoding, where draft and target models run on separate GPUs and exchange tokens over the network. The repository has two parts:

- **Simulator** (`src/`): a SimPy discrete-event simulator of draft/target pipelines, covering network topology, routing, batching, and acceptance replay.
- **End-to-end system** (`dspec_real/`): a real implementation on vLLM 0.27.1. Remote drafters connect to the target through vLLM's `custom_class` speculative hook, over ZMQ or HTTP.

The two share traces and calibration (`sim/`), so the same workload can run on real GPUs and in simulation.

## Layout

```
src/            simulator (sim.py, network/, performance/, acceptance/, trace/)
dspec_real/        end-to-end system (proto/, draft/, target/, policy/, router/, metrics/, experiments/)
sim/            sim configs from real runs, calibration fitting, real-vs-sim runs
scripts/        profiling, trace export, verification, cluster launchers
bench/          communication-latency and end-to-end campaign harnesses
experiments/    simulator benchmarks and configs
results/        checked-in result tables
docs/           design notes and result write-ups
aws-trn/        original gRPC prototype on AWS Trainium
```

## Simulator

```bash
pip install simpy pyyaml networkx
cd src && python sim.py --config ../experiments/configs/fat_tree_test.yaml
```

Acceptance is replayed from hardware traces when they are available. Otherwise it falls back to a `.joblib` predictor or a fixed rate.

- **Chain (SpecDec++ / vanilla SD):** each request carries `acceptance_seq`, a flat 0/1 list in draft order. Build it with `scripts/export_acceptance_seq_from_details.py`.
- **Tree (EAGLE, `framework: eagle`):** each request carries `tree_accept`, one entry per verification round:

  ```json
  {"round": 0, "depth": 4, "tree_nodes": 16, "accepted_path_len": 3}
  ```

  Each round advances the output by `accepted_path_len + 1`, and the target verifies `tree_nodes` candidates in one batch job. When `tree_nodes` is missing, `verified` and then `candidates` are used instead. Generate the traces with `scripts/profile_eagle_tree_accept.py`.

EAGLE hidden-state traffic is configured under `speculation.eagle`:

| Key | Default | Meaning |
|---|---|---|
| `hidden_dim` | 4096 | Target hidden-state width |
| `hidden_state_bytes_per_value` | 2.0 | Bytes per value |
| `tree_index_bytes_per_node` | 4.0 | Tree-structure bytes per candidate |
| `feature_init_base_ms`, `feature_init_ms_per_token` | 0.0 | Draft feature init, charged once before the first tree |

The prefill reply carries the prompt's hidden states, and each verify reply carries the hidden states of the committed tokens. These bytes only add latency when `network.bandwidth_mbps` is set.

Other features include fat-tree latency (`src/network/fat_tree_config.json`), routers (JSQ, Random, Round-Robin, JSQ(d), JIQ, and others), and length-aware batching (`queue_policy: lab`). See [docs/DESIGN.md](docs/DESIGN.md).

## End-to-end system

```bash
source scripts/env.sh
pytest tests -q                                   # CPU-only: transport, proposer, routing

python scripts/run_draft_server.py --drafter-id d0 --bind tcp://0.0.0.0:5560
```

On the target side, configure the proposer before constructing `LLM`:

```python
from dspec_real.proto.transport import make_client
from dspec_real.target import proposer

proposer.configure(make_client("zmq", addresses={"d0": "tcp://node1:5560"}))
target = LLM(model=..., speculative_config={
    "method": "custom_class",
    "model": "dspec_real.target.proposer.RemoteDraftProposer",
    "num_speculative_tokens": 4,
})
```

The control loop runs on the target: each engine step calls `propose()`, which fans out to all drafters and blocks until they reply. Decoding is greedy, so speculative output must match the non-speculative baseline token for token, and `tests/test_transport_parity.py` enforces this across transports.

Online experiments use `dspec_real/experiments/driver.py`, which writes `dsd_metrics.json` and `steps.jsonl`. `sim/calibrate_from_run.py` fits the simulator to those runs.

- Real-vs-sim and Table 3 results: [docs/table3-results.md](docs/table3-results.md)
- Full design: [plan.md](plan.md)
- vLLM hook findings: [docs/stage0-findings.md](docs/stage0-findings.md)
