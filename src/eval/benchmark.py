from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from rich.console import Console
from rich.table import Table

from src.eval.metrics import score

console = Console()


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def run_single(ref_path: Path, hyp_path: Path) -> Tuple[float, float]:
    ref = load_text(ref_path)
    hyp = load_text(hyp_path)
    return score(ref, hyp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple WER/CER benchmark")
    parser.add_argument("--ref", required=True, type=str, help="Reference text file or directory")
    parser.add_argument("--hyp", required=True, type=str, help="Hypothesis text file or directory")
    args = parser.parse_args()

    ref_path = Path(args.ref)
    hyp_path = Path(args.hyp)

    pairs: List[Tuple[Path, Path]] = []
    if ref_path.is_file() and hyp_path.is_file():
        pairs = [(ref_path, hyp_path)]
    else:
        ref_files = {p.stem: p for p in ref_path.glob("*.txt")}
        hyp_files = {p.stem: p for p in hyp_path.glob("*.txt")}
        common = set(ref_files.keys()) & set(hyp_files.keys())
        pairs = [(ref_files[k], hyp_files[k]) for k in sorted(common)]

    if not pairs:
        raise FileNotFoundError("No matching ref/hyp pairs found")

    table = Table(title="Benchmark")
    table.add_column("Sample")
    table.add_column("CER")
    table.add_column("WER")

    cer_sum = 0.0
    wer_sum = 0.0
    for ref, hyp in pairs:
        cer_val, wer_val = run_single(ref, hyp)
        cer_sum += cer_val
        wer_sum += wer_val
        table.add_row(ref.stem, f"{cer_val:.4f}", f"{wer_val:.4f}")

    total = len(pairs)
    table.add_row("AVERAGE", f"{cer_sum/total:.4f}", f"{wer_sum/total:.4f}")
    console.print(table)


if __name__ == "__main__":
    main()

