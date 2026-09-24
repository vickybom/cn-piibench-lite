"""PII-Auditor-CN-Lite: external black-box PII memorization audit toolkit.

Implements the CN-PIIBench-Lite evaluation pipeline as six modules:

    M1  m1_generator   Synthetic PII Generator          (before the model)
    M2  m2_prompts     Prompt-Matrix Builder            (before the model)
    M3  m3_inference   Inference Controller             (around the model)
    M4  m4_detector    Leakage Detector                 (after the model)
    M5  m5_metrics     Metric Engine (MER/CLMD/RW-MER)  (after the model)
    M6  m6_report      Risk Classifier + Reporter       (after the model)

The toolkit never accesses or modifies model weights, gradients, or training
data; it wraps the target LLM (before + after, never inside).
"""
__version__ = "1.0.0"
