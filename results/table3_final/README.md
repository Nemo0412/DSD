Compact copy of `runs/table3_final/` (runs/ is gitignored): the final small-scale
Table 3 on NYULAB and the dsdSim comparison produced locally by
`scripts/final_compare.sh`. Discussion: [docs/table3-results.md](../../docs/table3-results.md).

| file | what |
|---|---|
| `table3.md` | real Table 3, dsdSim and standard metric conventions |
| `real_vs_sim_pooled.md` / `real_vs_sim_tiered.md` | real vs sim, draft latency calibrated pooled / per N (tier) |
| `real_system_metrics.json` | system metrics + key config of every real run |
| `sim_tables.json` | sim TTFT / TPOT / throughput / acceptance per (method, N), both variants |
| `calibration.*.json` | sim latency constants fitted from the real runs (`*.fit.json`: fit quality) |
| `specpp_acceptance_profile.json` | per-position acceptance from the gamma=8 calibration run (SpecDec++ input, sim acceptance for vanilla/SpecDec++) |
| `eagle3_profile.json` | same, from the EAGLE3 runs (sim acceptance for the EAGLE3 row) |
