from __future__ import annotations

from typing import Tuple


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr_row = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr_row[j - 1] + 1
            delete = prev_row[j] + 1
            sub = prev_row[j - 1] + (ca != cb)
            curr_row.append(min(ins, delete, sub))
        prev_row = curr_row
    return prev_row[-1]


def cer(ref: str, hyp: str) -> float:
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return levenshtein(ref, hyp) / len(ref)


def wer(ref: str, hyp: str) -> float:
    ref_words = ref.split()
    hyp_words = hyp.split()
    ref_join = " ".join(ref_words)
    hyp_join = " ".join(hyp_words)
    return cer(ref_join, hyp_join)


def score(ref: str, hyp: str) -> Tuple[float, float]:
    return cer(ref, hyp), wer(ref, hyp)

