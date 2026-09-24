# Getting the real Qwen-2.5 results on a cloud GPU

The paper's primary models — **Qwen2.5-1.5B and 3B** — are too small to be hosted
by any commercial API (verified: they return 404 on DashScope), so they must run
as local weights on a GPU. Use a free / cheap cloud GPU.

The audit is *inference-only* (no training), and generation is batched, so a
single 16 GB GPU (Colab T4) handles all three scales under 4-bit quantization.

**Scale (matches the proposal).** `--n-entries` is the TOTAL number of synthetic
entries **across the seven categories**, so `--n-entries 1000` ≈ 140 per
category ≈ **10,000 queries per model** — the standard configuration in
Chapter 3.

---

## Path A — Google Colab (recommended, free tier works)

1. Build the upload bundle locally: `python make_package_zip.py`
   → creates `dist/pii_auditor_pkg.zip`.
2. Open <https://colab.research.google.com> → **File → Upload notebook** →
   `notebooks/CN_PIIBench_Lite_Colab.ipynb`.
3. **Runtime → Change runtime type → T4 GPU**.
4. Run the cells top to bottom; upload `pii_auditor_pkg.zip` when asked.
5. Pilot first (~15–30 min), then the full run (~1.5–3 h).
6. The last cell downloads `results_gpu.zip`. **Send it back** to have the real
   numbers written into Chapters 4 and 5.

## Path B — AutoDL (rent by the hour; better connectivity in China)

1. Rent an instance with any NVIDIA GPU ≥ 16 GB (RTX 3090/4090) + PyTorch image.
2. Upload `dist/pii_auditor_pkg.zip`, then:
   ```bash
   unzip pii_auditor_pkg.zip -d pii-auditor && cd pii-auditor
   pip install -r requirements-gpu.txt
   export HF_ENDPOINT=https://hf-mirror.com     # faster weight download in China
   python run_gpu.py --models Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B --n-entries 350   # pilot
   python run_gpu.py --full                                                        # 1.5B/3B/7B, 1,000 entries
   ```
3. Download the `results_gpu/` folder and send it back.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | `--batch-size 8` (default 32). 7B is the tight one. |
| Colab disconnects mid-run | Keep the tab visible; re-run — the corpus is regenerated from the fixed seed, so the run is reproducible. |
| Very slow weight download | Colab: just wait (one-time). AutoDL: set `HF_ENDPOINT=https://hf-mirror.com`. |
| `nvidia-smi` shows nothing | Runtime → Change runtime type → T4 GPU, then re-run from Step 1. |

## Notes

- **Base weights, not `-Instruct`.** `run_gpu.py` uses `Qwen/Qwen2.5-1.5B` etc.,
  because the study measures memorization of *pre-trained* weights via prefix /
  pre-query completion. This is deliberate and matches Chapter 3.
- **Greedy decoding** (temperature 0, `do_sample=False`) throughout, so a re-run
  reproduces the reported MER exactly.
- **Reproducibility.** The corpus regenerates from seed 20260524, identical to
  the one described in Chapter 3.
