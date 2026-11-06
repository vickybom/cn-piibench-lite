# Changelog

## 1.1 (5 November 2026)

Documentation and language only. No prompt, corpus value, record, metric, adapter or code path changed.

Added:

- `CODEBOOK.md`: every column and code in the records, and the factors they stand for.
- `LANGUAGE_NOTE.md`: where Chinese remains, why, and the English translation of every Chinese line in the code.

Changed in comments and report labels:

- `capability_eval.py:155`: the comment now gives the English meaning of the quoted word.
- `pii_auditor/finetune.py:24`: the docstring now translates its quoted example.
- `pii_auditor/m4_detector.py:15`: the module docstring now says that full names are matched exactly, as the code does. It used to describe a fuzzy match that was evaluated and rejected.
- `pii_auditor/m4_detector.py:60-61`: the docstring now romanizes and translates its examples.
- `pii_auditor/m4_detector.py:141`: the comment now translates its example.
- `pii_auditor/dashboard.py:178`: the dashboard stamp is now in English only.
- `notebooks/CN_PIIBench_Lite_Colab.ipynb`: the expected-result text now matches the dashboard stamp, in English.
- `piibench_demo.zip`: its copy of `pii_auditor/m4_detector.py` has the same comment changes.

Changed in `README.md` and `CITATION.cff`:

- The detector is described as exact match with checksum re-validation, as the code does.
- The aggregate RW-MER is described as an unweighted mean over the seven categories, as the code does.
- The first author is written Ming Mo (family name Mo) throughout.
- The archive is cited by its concept DOI, 10.5281/zenodo.22928924, which resolves to the latest version.
