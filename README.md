# Where Do Continual Forecasters Forget? Spectral Intrusion, Not Overlap, Explains Forgetting in Time-Series Forecasting

Code, configurations and result files for the paper of the same title (manuscript under review).

A forecaster that learns a sequence of regimes (seasons, sites, operating modes) forgets earlier ones. This
work measures **where in the frequency domain** that forgetting happens and finds that the intuitive answer
is wrong:

- **Overlap protects.** Forgetting *decreases* as the spectral overlap between an old and a new regime grows
  (Spearman *r* from −0.77 to −0.95 on linear, MLP and Mamba forecasters).
- **Intrusion forgets.** Forecast errors concentrate at frequencies that only the new regime contains: its
  oscillations *intrude* into old forecasts. On real benchmarks, overlap predicts where forgetting occurs in
  0 of 15 settings and intrusion in 9 of 15.
- **Theory.** For linear forecasters, the increase of an old regime's error decomposes exactly over
  frequency, ΔMSE = ∫ |H_new − H_old|² S_old dω (verified within 3%).
- **Remedy.** Intrusion-aware frequency rehearsal (IFR) weights the replay error by the frequencies a new
  regime threatens. It cuts forgetting by 42% in a controlled intervention and improves standard replay on
  all eight real benchmarks at no extra cost.

## Repository layout

```
run_continual.py         continual training and evaluation of all methods (one entry point)
spectracl/               forecasters (Mamba, linear, CLFD encoder) and continual-learning methods
  continual/intrusion_rehearsal.py   IFR, its ablations and the FreDF loss
  continual/                         EWC, SI, MAS, A-GEM, ER, MIR, DER++, CLS-ER, AdaptiveCL, FSNet, SpectraCL-v1
data/                    dataset utilities; put the CSV files here (see data/README.md)
configs/                 within-dataset (five seasonal regimes) and cross-domain (three source orders) configs
experiments/             synthetic studies, frequency-resolved analysis, Proposition 1 check, tables, figures
scripts/                 one script per experiment group (see below)
results/                 result files of every run reported in the paper (JSON)
```

## Installation

Python 3.10 and an NVIDIA GPU with CUDA 12 are required; the Mamba backbone uses the official CUDA
kernels.

```bash
pip install -r requirements.txt
```

If `mamba-ssm` or `causal-conv1d` fail to build, install the prebuilt wheels that match your PyTorch and
CUDA versions from their GitHub releases (<https://github.com/state-spaces/mamba/releases>,
<https://github.com/Dao-AILab/causal-conv1d/releases>).

## Data

Download the six public CSV files (Electricity, Traffic, ETTm1, ETTm2, ETTh1, Weather) into `data/`, as
described in [`data/README.md`](data/README.md).

## Quick start

```bash
# IFR and standard replay (ER) on cross-domain order A, one seed
python run_continual.py --config configs/crossdomain/crossdomain_A.yaml --data data \
    --methods ifr ifr_uniform --epochs 20 --seeds 0 --out_dir results/demo
```

Every run writes `continual_results_*.json` with the full results matrix (test MSE and MAE on every regime
after every regime), continual-learning metrics and an efficiency profile. Method names: `finetune`, `ewc`,
`si`, `mas`, `agem`, `ifr_uniform` (ER), `mir`, `derpp`, `clser`, `adaptivecl`, `fsnet`, `fredf`, `clfd`,
`clfd_ifr`, `spectracl` (SpectraCL-v1), `ifr`, the IFR ablations `ifr_shuffled`, `ifr_pooled`,
`ifr_neglect`, `ifr_overlap` and `ifr_old_energy`, and the output-gate variants `sg_ft`, `sg_er`, `sg_ifr`,
`clfd_sg_ifr` discussed in the paper. `--holdout val` evaluates on a validation split carved from the
training data, so that the test data are never seen during model selection.

## Reproducing the paper

All scripts run from the repository root; set `PYTHON` to choose the interpreter.

| Paper content | Command | Results |
|---|---|---|
| Overlap study, error growth by bin type, intervention study, Proposition 1 check | `bash scripts/run_synthetic.sh` | `results/synthetic_*.json`, `results/int1024_*.json` |
| Frequency-resolved forgetting on real data (three settings) | `bash scripts/run_real_band.sh mamba48` (and `linear48`, `mamba96`) | `results/real_band*/` |
| 16-method benchmark | `bash scripts/run_benchmark.sh all` | `results/main/`, `results/crossdomain/` |
| IFR ablation | `bash scripts/run_benchmark.sh all ifr_shuffled ifr_pooled ifr_neglect ifr_overlap ifr_old_energy` | as above |
| Sensitivity to κ and λ (validation) | `bash scripts/run_sensitivity.sh A` | `results/tuning/` |
| Output gate (validation, Discussion) | `bash scripts/run_validation.sh all sg_ft sg_er sg_ifr clfd_sg_ifr ifr ifr_uniform clfd clfd_ifr` | `results/val/` |
| Computational efficiency | `bash scripts/run_profile.sh` (on an otherwise idle GPU) | `results/profile/` |

Some within-dataset runs of the benchmark were made in an earlier batch and are stored in
`results/pilot/`; the table scripts merge `results/pilot/` and `results/main/` as in the paper.

**Tables and figures** are regenerated from the stored results without any training:

```bash
bash scripts/make_outputs.sh      # writes outputs/tables/*.tex and outputs/figs/*.pdf
```

| Output | Script |
|---|---|
| `fig0_concept`, `fig1_example`, `fig2_overlap`, `fig3_bins`, `fig4_real`, `fig5_intervention`, `tab_generality` | `experiments/make_paper_figures.py` |
| `tab_within`, `tab_within_fgt`, `tab_cross`, `tab_paired`, `tab_overall`, `tab_ablation`, `tab_efficiency` | `experiments/make_tables.py` |
| `tab_sensitivity` | `experiments/analyze_sensitivity.py` |
| `tab_prop` | `experiments/verify_proposition.py` |
| `tab_regimes`, gain vs. intrusion (needs the CSV files) | `experiments/make_regime_table.py`, `experiments/gain_vs_discrepancy.py` |
| statistics quoted in the text | `experiments/analyze_synthetic.py`, `analyze_intervention.py`, `analyze_real_band.py`, `pooled_test.py`, `compare_val.py` |

Results are averaged over five seeds within a dataset and ten across domains; all methods share each seed's
data order and initialisation, so differences are tested with paired tests. GPU kernels are not fully
deterministic, so reruns reproduce the reported numbers up to small run-to-run differences.

## Citation

```bibtex
@unpublished{suleiman2026intrusion,
  title  = {Where Do Continual Forecasters Forget? Spectral Intrusion, Not Overlap, Explains Forgetting in Time-Series Forecasting},
  author = {Suleiman, Aminu Bashir and Ran, Haowei and Ibrahim, Hamza and Garba, Ahmad Tijjani and Sun, Meijun and Wang, Zheng},
  note   = {Manuscript under review},
  year   = {2026}
}
```

## License

MIT; see [LICENSE](LICENSE). The datasets keep their own licenses.
