from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import typer
import ujson as json


app = typer.Typer(add_completion=False)


Lemma = Tuple[str, str, int]  # (short_name, fully_qualified, arity)


def lemma_families() -> List[Lemma]:
    return [
        ("add_comm", "Nat.add_comm", 2),
        ("mul_comm", "Nat.mul_comm", 2),
        ("add_assoc", "Nat.add_assoc", 3),
        ("mul_assoc", "Nat.mul_assoc", 3),
        ("add_zero", "Nat.add_zero", 2),
        ("zero_add", "Nat.zero_add", 2),
        ("mul_one", "Nat.mul_one", 2),
        ("one_mul", "Nat.one_mul", 2),
        # A few non-Nat toys
        ("and_comm", "Bool.and_comm", 2),
        ("or_comm", "Bool.or_comm", 2),
        ("append_assoc", "List.append_assoc", 3),
    ]


def variants_rw(lemma: str, include_synonyms: bool = True) -> List[str]:
    opts = [f"rw [{lemma}]", f"rw {lemma}"]
    if include_synonyms:
        opts.append(f"rewrite {lemma}")
    return opts


def variants_refl() -> List[str]:
    return ["refl", "rfl"]


def make_example(example_id: str, lemma: Lemma, synonym_rate: float, rng: random.Random) -> Dict[str, Union[str, List[Union[str, List[str]]]]]:
    short, fq, arity = lemma
    steps: List[Union[str, List[str]]] = ["intro a", "intro b"]
    if arity >= 3:
        steps.append("intro c")

    # Identity-type lemmas solved by simp
    if short in {"add_zero", "zero_add", "mul_one", "one_mul"}:
        steps.append("simp")
    elif short in {"and_comm", "or_comm"}:  # treat as commutativity
        steps.append(variants_rw(fq, include_synonyms=rng.random() < synonym_rate))
        steps.append(rng.choice(variants_refl()))
    elif short == "append_assoc":
        steps.append(variants_rw(fq, include_synonyms=rng.random() < synonym_rate))
        steps.append(rng.choice(variants_refl()))
    else:
        steps.append(variants_rw(fq, include_synonyms=rng.random() < synonym_rate))
        steps.append(rng.choice(variants_refl()))

    return {"theorem_id": example_id, "oracle_steps": steps}


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r))
            f.write("\n")


@app.command()
def main(
    out_dir: Path = typer.Option(Path("data/large"), help="Output directory"),
    n_examples: int = typer.Option(10000, help="Total examples to generate"),
    train_ratio: float = typer.Option(0.8, help="Train split ratio"),
    val_ratio: float = typer.Option(0.1, help="Val split ratio (rest is test)"),
    seed: int = typer.Option(7, help="Random seed"),
    synonym_rate: float = typer.Option(0.5, help="Probability to include extra rw synonyms"),
    families: List[str] = typer.Option([], help="Subset of lemma families to use (default: all)"),
):
    rng = random.Random(seed)

    lemmas = lemma_families()
    if families:
        famset = set(families)
        lemmas = [x for x in lemmas if x[0] in famset]
        if not lemmas:
            raise typer.BadParameter("No matching families found.")

    # Generate examples uniformly across selected families
    rows: List[Dict] = []
    for i in range(n_examples):
        lemma = lemmas[i % len(lemmas)]
        ex = make_example(f"ex_{lemma[0]}_{i}", lemma, synonym_rate, rng)
        rows.append(ex)

    # Shuffle & split
    rng.shuffle(rows)
    n_train = int(len(rows) * train_ratio)
    n_val = int(len(rows) * val_ratio)
    train = rows[:n_train]
    val = rows[n_train : n_train + n_val]
    test = rows[n_train + n_val :]

    # Write shards
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    write_jsonl(out_dir / "test.jsonl", test)

    # Manifest
    manifest = {
        "n_examples": len(rows),
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "families": [x[0] for x in lemmas],
        "seed": seed,
        "synonym_rate": synonym_rate,
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote dataset to {out_dir} :: {manifest}")


if __name__ == "__main__":
    app()

