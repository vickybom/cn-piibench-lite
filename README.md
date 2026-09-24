# PII-Auditor-CN-Lite

An **external, black-box audit toolkit** that measures Chinese PII memorization
and cross-lingual leakage in small & medium domestic LLMs — the applied
deliverable of the *CN-PIIBench-Lite* dissertation. It wraps a target model
(before + after) and **never accesses or modifies model weights**.

```
Toolkit (M1–M2)  →  prompts  →  LLM (M3)  →  outputs  →  Toolkit (M4–M6)
```

| Module | File | Role | Position |
|--------|------|------|----------|
| M1 | `pii_auditor/m1_generator.py` | Synthetic PII Generator (7 categories, real checksums) | before |
| M2 | `pii_auditor/m2_prompts.py` | Prompt-Matrix Builder (Type A/B/C/D, 12 templates) | before |
| M3 | `pii_auditor/m3_inference.py` | Inference Controller (greedy decoding; demo/API/local) | around |
| M4 | `pii_auditor/m4_detector.py` | Leakage Detector (exact + fuzzy + checksum re-validation) | after |
| M5 | `pii_auditor/m5_metrics.py` | Metric Engine (MER, CLMD, RW-MER) | after |
| M6 | `pii_auditor/m6_report.py` | Risk Classifier + Reporter (Low/Medium/High) | after |

## Install

```bash
pip install -r requirements.txt
```

## Quick start (simulation — runs anywhere, no GPU, no key)

```bash
python cli.py audit --backend demo \
  --models qwen2.5-1.5b qwen2.5-3b qwen2.5-7b --n-entries 1000
```

Outputs land in `results/`:
`summary.csv`, `per_category.csv`, `mer_long.csv`, `results.json`,
`audit_report.html`, and **`dashboard.html`** (double-click for the defense demo).

> The `demo` backend is a **clearly-labelled simulation** used to exercise the
> pipeline and the dashboard. Its numbers are illustrative, not real
> measurements. Use a real backend (below) for dissertation results.

## Real run — API (QwenCloud / DashScope, no GPU needed)

1. Copy `.env.example` to `.env` and paste your key (see `.env` comments).
2. Discover which model IDs your account serves:
   ```bash
   python test_connection.py
   ```
3. Run a pilot:
   ```bash
   python cli.py audit --backend dashscope \
     --models qwen2.5-1.5b-instruct qwen2.5-3b-instruct qwen2.5-7b-instruct \
     --n-entries 30 --max-queries-per-model 400
   ```

## Real run — local base weights (GPU workstation, primary dissertation path)

Uncomment `transformers`/`torch` in `requirements.txt`, then:

```bash
python cli.py audit --backend hf_local \
  --models Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B --n-entries 1000
```

## Metrics

- **MER** = fraction of synthetic entries reproduced verbatim/near-verbatim
  under greedy decoding, per (model, category, probing condition).
- **CLMD** = MER(EN→ZH) − MER(ZH→ZH), taken **between matched template pairs
  only** (C1/D1 and C2/D2), so that the subtraction changes the language of the
  instruction and nothing else; positive ⇒ cross-lingual safety gap. The
  unpaired family-mean version is still computed for comparison and is
  reported as misleading in the paper.
- **RW-MER** = MER × PRI(category); aggregate = PRI-weighted mean → Low/Medium/High.

## Ethics

All PII is synthetic and fictitious. Every value is generated from its national
format rule under a fixed seed and validated against the governing standard; no
value was sampled from any real or public source, and no real personal data is
collected, stored or processed at any stage.

## Data archive

The code in this repository reproduces the analysis; the records it analyses are
archived separately, because they are about 565 MB:

> Mo Ming and Davood Pour Yousefian Barfeh (2026). *CN-PIIBench-Lite: toolkit,
> synthetic corpus generator, probe matrix, per-query records and adapters.*
> Zenodo. https://doi.org/10.5281/zenodo.22928925

The deposit holds the corpus generator and its seed, the twelve-template probe
matrix, the retained per-query records of every condition (414,720 probes), the
result bundles, the LoRA adapters of the audited checkpoints, `ARTIFACT_HASHES.txt`
(SHA-256 of all 167 record files, 2.17 GB uncompressed) and `MODEL_REVISIONS.txt`.
Download the bundles next to this repository and every table and figure of the
paper recomputes offline, without a GPU.

## Citation

```bibtex
@article{ming2026cnpiibench,
  author  = {Ming, Mo and Pour Yousefian Barfeh, Davood},
  title   = {{CN-PIIBench-Lite}: Measuring {Chinese} {PII} Memorization and
             Cross-Lingual Leakage in Small and Medium Domestic Language Models},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## License

Code is released under the MIT License (`LICENSE`). The synthetic corpus, the
records and the result bundles in the Zenodo deposit are released under
CC BY 4.0 (`LICENSE-DATA`). The model weights the study audits are not
redistributed; only the LoRA adapters trained in this work are.
