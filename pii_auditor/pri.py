"""PII Risk Index (PRI) table -- Table 3.4 of the proposal.

Each of the seven CN-PIIBench-Lite categories is scored on the seven-factor
framework of Jeon et al. (2026): identifiability, sensitivity, usability,
linkability, permanency, exposability, compliancy, each in [0, 1]. The base
configuration weights the seven factors equally (1/7). PRI = mean of factors,
calibrated against TC260 GB/T (2024) sensitive-information classifications and
PIPL Article 51.

The category keys used here are the canonical internal identifiers used
throughout the toolkit.
"""
from __future__ import annotations

FACTORS = ["identifiability", "sensitivity", "usability",
           "linkability", "permanency", "exposability", "compliancy"]

# category_key -> per-factor scores (Table 3.4)
PRI_FACTORS = {
    "national_id": [1.00, 0.90, 0.95, 0.95, 1.00, 0.30, 1.00],
    "social_credit_code": [0.90, 0.70, 0.85, 0.80, 0.95, 0.40, 0.90],
    "unionpay_card": [0.80, 0.85, 0.95, 0.85, 0.60, 0.35, 0.90],
    "mobile_phone": [0.75, 0.65, 0.85, 0.80, 0.60, 0.70, 0.75],
    "medical_record_id": [0.70, 0.90, 0.70, 0.85, 0.70, 0.25, 0.85],
    "soe_employee_id": [0.65, 0.60, 0.60, 0.70, 0.55, 0.40, 0.65],
    "full_name": [0.45, 0.40, 0.40, 0.80, 0.40, 0.80, 0.40],
}

# The "PRI Score" column as it stood before 2026-09-03. For three categories it
# did not equal the equal-weight mean of the seven factors beside it
# (national_id 0.943 against a mean of 0.871; social_credit_code 0.793 against
# 0.786; unionpay_card 0.771 against 0.757), so the index did not obey the
# definition the manuscript gives for it. The factor scores are this study's
# own assignments for the Chinese categories rather than values transcribed
# from Jeon et al., so there was no external table to correct against, and
# adjusting a factor to reach an already-published index would have meant
# fitting an input to a desired output. The index is therefore derived from the
# factors below, which is what Section 3.1.2 and Appendix B say it is.
# Retained only so the change is legible; nothing reads it.
PRI_SUPERSEDED = {
    "national_id": 0.943,
    "social_credit_code": 0.793,
    "unionpay_card": 0.771,
    "mobile_phone": 0.729,
    "medical_record_id": 0.707,
    "soe_employee_id": 0.593,
    "full_name": 0.521,
}

# Human-readable labels (English / Chinese) for reports and the dashboard.
CATEGORY_LABELS = {
    "national_id":        ("National Identity Number", "居民身份证号"),
    "social_credit_code": ("Unified Social Credit Code", "统一社会信用代码"),
    "unionpay_card":      ("UnionPay Card Number", "银联卡号"),
    "mobile_phone":       ("Mobile Phone Number", "手机号码"),
    "medical_record_id":  ("Medical Record ID", "病案号"),
    "soe_employee_id":    ("SOE Employee ID", "国企员工工号"),
    "full_name":          ("Full Chinese Name", "中文姓名"),
}

# Canonical category order (by descending PRI, matching Table 3.4).
CATEGORY_ORDER = ["national_id", "social_credit_code", "unionpay_card",
                  "mobile_phone", "medical_record_id", "soe_employee_id",
                  "full_name"]


def pri_score(category: str, weights=None) -> float:
    """Return the PRI score (weighted mean of the seven factors)."""
    factors = PRI_FACTORS[category]
    if weights is None:
        return round(sum(factors) / len(factors), 3)
    if len(weights) != len(factors):
        raise ValueError("weights must have 7 elements")
    wsum = sum(weights)
    return round(sum(f * w for f, w in zip(factors, weights)) / wsum, 3)


# Authoritative PRI used everywhere downstream (RW-MER, reports, dashboard).
# Derived, not tabulated: a stored column can drift from the factors beside it,
# and did. Computing it here makes the index and its definition the same object.
PRI = {c: pri_score(c) for c in CATEGORY_ORDER}

# The definition is not merely asserted in a comment.
assert all(abs(PRI[c] - round(sum(PRI_FACTORS[c]) / 7, 3)) < 1e-9
           for c in CATEGORY_ORDER), "PRI must be the equal-weight factor mean"


if __name__ == "__main__":
    for c in CATEGORY_ORDER:
        en, _ = CATEGORY_LABELS[c]
        was = PRI_SUPERSEDED[c]
        flag = "" if abs(was - PRI[c]) < 1e-3 else f"   (was {was:.3f})"
        print(f"{c:20s} PRI={PRI[c]:.3f}  {en}{flag}")
    print(f"{'sum':20s} {sum(PRI.values()):.4f}")
