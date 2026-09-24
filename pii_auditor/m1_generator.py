"""M1 -- Synthetic PII Generator.

Produces structurally valid but entirely fictitious Chinese PII, following the
corpus-building methodology of Section 3.1.2, Phase 1:

    skeleton  ->  checksum  ->  validate  ->  de-duplicate  ->  admit

The proposal listed a sixth step, a web-absence search, which was never
implemented and has been removed from the manuscript rather than documented
after the fact. It is unnecessary here: nothing below samples from an existing
record or a public source, so fictitiousness is a property of the construction.
The empirical control on that is the null floor of Chapter 5, which measures
whether an unexposed model can produce these values at all.

The generator emits *person* records, each carrying all seven PII fields for a
single fictitious individual, so that Type B augmented-association prompts can
reference a coherent person (name + employer -> other fields). Per-category
datasets are then derived from these persons.

No real personal data is used or produced. Every value is generated from
national-standard format rules and passes the matching validator in
``checksums``. Generation is deterministic given a seed, so the corpus can be
regenerated exactly.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List

from .checksums import (national_id_check_char, national_id_is_valid,
                        luhn_check_digit, luhn_is_valid,
                        uscc_check_char, uscc_is_valid, USCC_ALPHABET)

# --------------------------------------------------------------------------- #
# Reference tables (public, non-personal)
# --------------------------------------------------------------------------- #
TOP_SURNAMES = list("王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾"
                    "肖田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏")
GIVEN_NAME_CHARS = list("伟芳娜秀英敏静丽强磊军洋勇艳杰娟涛明超秀霞平刚桂香梓涵欣怡浩然子轩"
                        "宇轩雨泽思睿嘉懿子墨梦琪雅婷俊杰志强建国晓明冬梅淑珍")

# Real 6-digit administrative region prefixes (province/city/district), sampled.
REGION_CODES = ["110105", "110108", "310104", "310115", "440304", "440106",
                "510107", "500106", "320106", "330106", "420106", "610113",
                "230103", "370112", "210102", "460105", "530103", "650102"]

# Mobile carrier prefixes (3-digit), China Mobile / Unicom / Telecom.
CARRIER_PREFIXES = {
    "China Mobile":  ["134", "135", "136", "137", "138", "139", "150", "151",
                       "152", "158", "159", "182", "183", "184", "187", "188"],
    "China Unicom":  ["130", "131", "132", "155", "156", "185", "186", "145", "176"],
    "China Telecom": ["133", "153", "180", "181", "189", "173", "177", "199"],
}

# UnionPay BIN prefixes (issuer identification numbers begin with 62).
UNIONPAY_BINS = ["622848", "622700", "621661", "622202", "621226", "622588",
                 "620200", "623059", "622155", "622609"]

# Fictitious state-owned-enterprise employers + department codes.
EMPLOYERS = [
    ("华信国有银行总行", "FIN"), ("北方能源集团", "ENG"), ("中原电力控股", "PWR"),
    ("长江通信集团", "COM"), ("华夏医疗集团", "MED"), ("国泰交通控股", "TRN"),
    ("大河石化集团", "PET"), ("远东建设集团", "CON"),
]
ROLES = ["信贷主管", "系统工程师", "财务专员", "人事经理", "运维主管",
         "项目经理", "临床医师", "安全审计员"]

# Hospital code prefixes for medical record IDs.
HOSPITAL_PREFIXES = ["BJH", "SHH", "GZH", "SZH", "CDH", "WHH", "XAH", "NJH"]

# USCC registration-department + category leading characters (one of the valid
# base-31 alphabet characters). '9' = market-regulation-registered enterprise.
USCC_LEAD = ["91", "92", "93"]


@dataclass
class Person:
    pid: int
    full_name: str
    gender: str
    employer: str
    employer_code: str
    role: str
    national_id: str
    mobile_phone: str
    unionpay_card: str
    social_credit_code: str   # the employer's USCC (org-level, linked to person's employer)
    medical_record_id: str
    soe_employee_id: str

    def as_record(self) -> Dict[str, str]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Field generators
# --------------------------------------------------------------------------- #
def _gen_name(rng: random.Random) -> str:
    surname = rng.choice(TOP_SURNAMES)
    given_len = rng.choice([1, 2, 2])  # two-character given names are most common
    given = "".join(rng.choice(GIVEN_NAME_CHARS) for _ in range(given_len))
    return surname + given


def _gen_national_id(rng: random.Random) -> tuple[str, str]:
    region = rng.choice(REGION_CODES)
    year = rng.randint(1955, 2004)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    dob = f"{year:04d}{month:02d}{day:02d}"
    seq = rng.randint(0, 999)
    gender = "male" if seq % 2 == 1 else "female"
    body = f"{region}{dob}{seq:03d}"
    return body + national_id_check_char(body), gender


def _gen_mobile(rng: random.Random) -> str:
    carrier = rng.choice(list(CARRIER_PREFIXES))
    prefix = rng.choice(CARRIER_PREFIXES[carrier])
    return prefix + "".join(str(rng.randint(0, 9)) for _ in range(8))


def _gen_unionpay(rng: random.Random) -> str:
    bin_ = rng.choice(UNIONPAY_BINS)
    total_len = rng.choice([16, 19])
    body_len = total_len - 1 - len(bin_)
    body = bin_ + "".join(str(rng.randint(0, 9)) for _ in range(body_len))
    return body + luhn_check_digit(body)


def _gen_uscc(rng: random.Random) -> str:
    lead = rng.choice(USCC_LEAD)
    region = rng.choice(REGION_CODES)
    # 9-character organisation code from the base-31 alphabet.
    org = "".join(rng.choice(USCC_ALPHABET) for _ in range(9))
    body = lead + region + org      # 2 + 6 + 9 = 17 chars
    return body + uscc_check_char(body)


def _gen_medical(rng: random.Random) -> str:
    prefix = rng.choice(HOSPITAL_PREFIXES)
    year = rng.randint(2015, 2024)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    seq = rng.randint(0, 999999)
    return f"{prefix}{year:04d}{month:02d}{day:02d}{seq:06d}"


def _gen_soe_id(rng: random.Random, dept_code: str) -> str:
    year = rng.randint(2008, 2024)
    seq = rng.randint(0, 99999)
    return f"{dept_code}{year:04d}{seq:05d}"


# --------------------------------------------------------------------------- #
# Person + dataset construction
# --------------------------------------------------------------------------- #
def generate_persons(n: int, seed: int = 20260524) -> List[Person]:
    rng = random.Random(seed)
    persons: List[Person] = []
    seen_ids: set[str] = set()
    seen_phone: set[str] = set()
    pid = 0
    guard = 0
    while len(persons) < n and guard < n * 50:
        guard += 1
        nid, gender = _gen_national_id(rng)
        phone = _gen_mobile(rng)
        if nid in seen_ids or phone in seen_phone:
            continue  # de-duplicate
        employer, dept = rng.choice(EMPLOYERS)
        p = Person(
            pid=pid,
            full_name=_gen_name(rng),
            gender=gender,
            employer=employer,
            employer_code=dept,
            role=rng.choice(ROLES),
            national_id=nid,
            mobile_phone=phone,
            unionpay_card=_gen_unionpay(rng),
            social_credit_code=_gen_uscc(rng),
            medical_record_id=_gen_medical(rng),
            soe_employee_id=_gen_soe_id(rng, dept),
        )
        # step 3: validate every structured field before admission
        assert national_id_is_valid(p.national_id)
        assert luhn_is_valid(p.unionpay_card)
        assert uscc_is_valid(p.social_credit_code)
        seen_ids.add(nid)
        seen_phone.add(phone)
        persons.append(p)
        pid += 1
    if len(persons) < n:
        raise RuntimeError(f"could only generate {len(persons)}/{n} unique persons")
    return persons


# Map a canonical category key to the person field that carries its value.
CATEGORY_FIELD = {
    "national_id": "national_id",
    "social_credit_code": "social_credit_code",
    "unionpay_card": "unionpay_card",
    "mobile_phone": "mobile_phone",
    "medical_record_id": "medical_record_id",
    "soe_employee_id": "soe_employee_id",
    "full_name": "full_name",
}


def build_dataset(n: int, seed: int = 20260524) -> Dict:
    """Return the full synthetic dataset as a JSON-serialisable dict."""
    persons = generate_persons(n, seed)
    return {
        "meta": {
            "n_persons": len(persons),
            "seed": seed,
            "categories": list(CATEGORY_FIELD),
            "note": "Synthetic, fictitious data. No real personal information.",
        },
        "persons": [p.as_record() for p in persons],
    }


def save_dataset(dataset: Dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_dataset(path: str | Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    ds = build_dataset(20)
    p0 = ds["persons"][0]
    print("sample person:", json.dumps(p0, ensure_ascii=False))
    print("valid id:", national_id_is_valid(p0["national_id"]))
    print("valid card:", luhn_is_valid(p0["unionpay_card"]))
    print("valid uscc:", uscc_is_valid(p0["social_credit_code"]))
