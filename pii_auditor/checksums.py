"""Checksum / format validators for Chinese structured identifiers.

These implement the real national-standard check algorithms cited in the
proposal (Table 3.1):

  * National Identity Number  -> ISO 7064:1983 MOD 11-2  (GB 11643-1999)
  * UnionPay Card Number      -> Luhn (ISO/IEC 7812)
  * Unified Social Credit Code -> GB 32100-2015 mod-31 weighted check

Every generator in ``m1_generator`` produces entries that pass the matching
validator here, and the leakage detector (M4) re-runs these validators on any
structured match so that a format-valid but wrong hallucinated identifier is
rejected rather than counted as a leak.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# National Identity Number (18 digit) -- ISO 7064 MOD 11-2, GB 11643-1999
# --------------------------------------------------------------------------- #
_ID_WEIGHTS = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
_ID_CHECK_MAP = ["1", "0", "X", "9", "8", "7", "6", "5", "4", "3", "2"]


def national_id_check_char(body17: str) -> str:
    """Return the ISO 7064 MOD 11-2 check character for the first 17 digits."""
    if len(body17) != 17 or not body17.isdigit():
        raise ValueError("national id body must be 17 digits")
    total = sum(int(d) * w for d, w in zip(body17, _ID_WEIGHTS))
    return _ID_CHECK_MAP[total % 11]


def national_id_is_valid(value: str) -> bool:
    value = value.strip().upper()
    if len(value) != 18 or not value[:17].isdigit():
        return False
    if value[17] not in "0123456789X":
        return False
    try:
        return national_id_check_char(value[:17]) == value[17]
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
# UnionPay Card Number (16-19 digit) -- Luhn / ISO-IEC 7812
# --------------------------------------------------------------------------- #
def luhn_check_digit(number_without_check: str) -> str:
    """Return the Luhn check digit that should be appended to ``number``."""
    digits = [int(d) for d in number_without_check]
    total = 0
    # The appended check digit is at an even position from the right (index 0),
    # so the right-most *body* digit is at an odd position and gets doubled.
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return str((10 - (total % 10)) % 10)


def luhn_is_valid(number: str) -> bool:
    number = number.strip()
    if not number.isdigit() or len(number) < 2:
        return False
    return luhn_check_digit(number[:-1]) == number[-1]


# --------------------------------------------------------------------------- #
# Unified Social Credit Code (18 char) -- GB 32100-2015
# --------------------------------------------------------------------------- #
# Base-31 alphabet: digits 0-9 and A-Z excluding I, O, S, V, Z.
USCC_ALPHABET = "0123456789ABCDEFGHJKLMNPQRTUWXY"
_USCC_WEIGHTS = [1, 3, 9, 27, 19, 26, 16, 17, 20, 29, 25, 13, 8, 24, 10, 30, 28]
_USCC_INDEX = {c: i for i, c in enumerate(USCC_ALPHABET)}


def uscc_check_char(body17: str) -> str:
    """Return the GB 32100-2015 check character for the first 17 characters."""
    if len(body17) != 17:
        raise ValueError("uscc body must be 17 characters")
    total = 0
    for c, w in zip(body17, _USCC_WEIGHTS):
        if c not in _USCC_INDEX:
            raise ValueError(f"illegal USCC character: {c!r}")
        total += _USCC_INDEX[c] * w
    c = 31 - (total % 31)
    if c == 31:
        c = 0
    return USCC_ALPHABET[c]


def uscc_is_valid(value: str) -> bool:
    value = value.strip().upper()
    if len(value) != 18:
        return False
    if any(c not in _USCC_INDEX for c in value):
        return False
    try:
        return uscc_check_char(value[:17]) == value[17]
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Classic public test vector for the Chinese resident ID (check char 'X').
    assert national_id_is_valid("11010519491231002X"), "known-valid ID rejected"

    # Round-trip: a correct check char validates, a wrong one does not.
    body = "11010519900307161"
    ck = national_id_check_char(body)
    assert national_id_is_valid(body + ck)
    wrong = "0" if ck != "0" else "1"
    assert not national_id_is_valid(body + wrong)

    up_body = "622536527156282"
    up = up_body + luhn_check_digit(up_body)
    assert luhn_is_valid(up)
    assert not luhn_is_valid(up[:-1] + ("0" if up[-1] != "0" else "1"))

    us_body = "91350100M000100Y4"
    us = us_body + uscc_check_char(us_body)
    assert uscc_is_valid(us), us
    assert not uscc_is_valid(us[:-1] + ("0" if us[-1] != "0" else "1"))

    print("checksums.py self-test passed")
