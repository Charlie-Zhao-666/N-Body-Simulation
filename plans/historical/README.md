# Historical experiment plans

These JSON files are directly loadable in **Test mode -> Load plan**. They reproduce the saved input parameters and comparison settings of the original batches, rather than reconstructing dt from rounded table cells. No simulations were rerun to create these files.

Line numbers refer to the published [runs.dat](../../data/run-history/runs.dat), including its header lines. New runs append new records; they do not reuse or overwrite the historical rows.

## Four main experiments

| Plan | Original lines | Runs | Purpose | Reference row |
| --- | --- | ---: | --- | ---: |
| [01: KDK dt scan](01_lines_36-44_kdk_dt_scan.json) | 36-44 | 9 | Fixed-step KDK across nine time steps | 36 |
| [02: method scan](02_lines_49-50_method_scan.json) | 49-50 | 2 | KDK versus RK4 at the same dt | 50 |
| [03: adaptive scan](03_lines_53-60_adaptive_scan.json) | 53-60 | 8 | Fine-step reference, fixed KDK, timescale eta=0.03/0.01, acceleration ratios (N,K)=(2,7),(3,4),(5,3),(10,2) | 53 |
| [04: cost comparison](04_lines_67-69_cost_comparison.json) | 67-69 | 3 | RK4 reference, KDK at the same dt, RK4 with dt multiplied by four | 67 |

The fourth experiment extends through **line 69**, not line 68: line 69 is the RK4 run with the four-times larger dt. The historical report's `67-68` label omitted this final member of the saved batch. The factor of four was a proposed cost matching assumption; it does not guarantee equal measured CPU cost.

All four use Butterfly I, CPU, T=6.2346748391 and diagnostics every 10 base steps. All position comparisons use 1001 uniform time samples with RMSE and NRMSE enabled. Full-precision values and inactive adaptive settings remain exactly as stored in the source batch plans.

## Additional experiments inside the requested 53-68 range

| Plan | Original lines | Runs | Purpose | Reference row |
| --- | --- | ---: | --- | ---: |
| [Supplement: first trial](supplement_lines_61-63_method_dt_trial.json) | 61-63 | 3 | RK4 reference, same-dt KDK, four-times-dt KDK | 61 |
| [Supplement: second trial](supplement_lines_64-66_method_dt_trial.json) | 64-66 | 3 | RK4 reference, same-dt KDK, four-times-dt RK4 | 64 |

These remain separate because each original batch had its own reference. Merging all of lines 53-68 into one plan would change the trajectory comparisons.

## Loading and interpretation

1. Download the repository and open `interactive_nbody_lab.py`.
2. Open **Test mode**, choose **Load plan**, and select one JSON file here. Loading fills the original parameters without starting a simulation.
3. Check the reference marker and parameters, then start the batch when ready. The current application runs the selected reference first. For lines 49-50 this means historical row 50 runs before row 49; the original comparison baseline is preserved.
4. Results are saved through the existing batch recording system in the script's `graph and data` directory. Actual output line numbers and timings depend on the new run.

These are historical reproductions, not certificates of exact trajectory accuracy. References are numerical trajectories. Current software and hardware can change runtime and numerical results.

The dt-scan reference has **8,000,000 steps** and the adaptive-scan reference has **16,000,000 steps**. The application retains every step, so these plans can require substantial RAM and disk space. A sampling interval of 10 reduces diagnostic sampling, not full trajectory retention. They have been validated for import; they were not rerun during publication.

## Source batches

- [01_lines_36-44_kdk_dt_scan.json](01_lines_36-44_kdk_dt_scan.json): [batch_2026-10-04_03-31-13_099261.json](../../data/run-history/batch_2026-10-04_03-31-13_099261.json).
- [02_lines_49-50_method_scan.json](02_lines_49-50_method_scan.json): [batch_2026-10-04_17-06-39_626746.json](../../data/run-history/batch_2026-10-04_17-06-39_626746.json).
- [03_lines_53-60_adaptive_scan.json](03_lines_53-60_adaptive_scan.json): [batch_2026-10-05_00-13-16_101253.json](../../data/run-history/batch_2026-10-05_00-13-16_101253.json).
- [04_lines_67-69_cost_comparison.json](04_lines_67-69_cost_comparison.json): [batch_2026-10-05_22-51-28_408049.json](../../data/run-history/batch_2026-10-05_22-51-28_408049.json).
- [supplement_lines_61-63_method_dt_trial.json](supplement_lines_61-63_method_dt_trial.json): [batch_2026-10-05_22-46-46_596541.json](../../data/run-history/batch_2026-10-05_22-46-46_596541.json).
- [supplement_lines_64-66_method_dt_trial.json](supplement_lines_64-66_method_dt_trial.json): [batch_2026-10-05_22-49-39_112215.json](../../data/run-history/batch_2026-10-05_22-49-39_112215.json).
