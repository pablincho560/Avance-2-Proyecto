"""Compara dos corridas de Ragas (antes/despues) y genera tabla + grafico.

Uso: python -m evaluation.compare_results --before baseline --after mejora
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RES = Path(__file__).parent / "results"


def load(tag: str) -> dict:
    return json.loads((RES / f"{tag}_resumen.json").read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="baseline")
    ap.add_argument("--after", default="mejora")
    args = ap.parse_args()

    b, a = load(args.before), load(args.after)
    df = pd.DataFrame({args.before: b["metricas"], args.after: a["metricas"]})
    df["delta"] = (df[args.after] - df[args.before]).round(4)

    print("\nConfig antes:", b["config"])
    print("Config despues:", a["config"])
    print("\n" + df.to_markdown() if hasattr(df, "to_markdown") else "\n" + df.to_string())

    ax = df[[args.before, args.after]].plot(kind="bar", figsize=(8, 5))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Puntaje (0-1)")
    ax.set_title("Métricas Ragas: antes vs. después de la mejora")
    plt.xticks(rotation=20)
    plt.tight_layout()
    out = RES / f"comparacion_{args.before}_vs_{args.after}.png"
    plt.savefig(out, dpi=150)
    print(f"\nGrafico guardado en {out}")


if __name__ == "__main__":
    main()
