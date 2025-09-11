from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple, Dict
import re

from dspy_lean_prover.eval_utils import matches_oracle, normalize_tactic
from dspy_lean_prover.hinting import HintMode
from dspy_lean_prover.modules import IterativeProver, SolveWithIterativeProver
from dspy_lean_prover.verifier import LeanVerifier


def load_dataset_any(path: Path) -> List[dict]:
    # Import lightweight JSON parser only when needed at runtime
    import ujson as json

    if path.is_dir():
        # Read all jsonl files under directory (train/val/test). Use all rows combined.
        rows: List[dict] = []
        jsonl_files = sorted(list(path.rglob("*.jsonl")))
        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files found under {path}")
        for p in jsonl_files:
            with p.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
        return rows

    # Single file: support .jsonl or .json
    if path.suffix == ".jsonl":
        rows: List[dict] = []
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    else:
        with path.open("r") as f:
            return json.load(f)


def split_dataset(examples: List[dict], seed: int, train_ratio: float) -> Tuple[List[dict], List[dict]]:
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    cut = int(len(idx) * train_ratio)
    train = [examples[i] for i in idx[:cut]]
    test = [examples[i] for i in idx[cut:]]
    return train, test


def make_examples(dicts: List[dict]) -> List[dspy.Example]:
    import dspy

    return [
        dspy.Example(
            theorem_id=d["theorem_id"], oracle_steps=d["oracle_steps"]
        ).with_inputs("theorem_id", "oracle_steps")
        for d in dicts
    ]


def select_curated(dicts: List[dict], max_len: int = 4) -> List[dict]:
    """Pick easier theorems to bootstrap compilation.

    Heuristic: total steps <= max_len and final step is 'refl' or 'simp'.
    """
    curated: List[dict] = []
    for d in dicts:
        steps = d["oracle_steps"]
        if len(steps) <= max_len:
            last = steps[-1]
            last_str = last[0] if isinstance(last, list) and last else last
            if isinstance(last_str, str) and (last_str.startswith("refl") or last_str.startswith("simp")):
                curated.append(d)
    return curated


def kl_metric(example, prediction, trace=None, prompt_callback=None, kl_weight=0.0):
    import dspy
    import numpy as np
    from dspy_lean_prover.custom_lm import CustomLM

    accuracy = matches_oracle(prediction.proof, example.oracle_steps)

    if kl_weight == 0.0:
        return accuracy

    # Get the last response from the history
    last_response = dspy.settings.lm.history[-1] if dspy.settings.lm.history else None
    if not last_response:
        return accuracy

    # Get the log-probabilities from the compiled program
    compiled_logprobs = last_response["choices"][0].get("logprobs")

    # Get the prompt from the callback
    prompt = prompt_callback.prompts[-1] if prompt_callback.prompts else None
    if not prompt:
        return accuracy

    # Store the original LM
    original_lm = dspy.settings.lm

    # Create a reference model
    reference_lm = CustomLM(model=original_lm.model, logprobs=True)
    dspy.settings.configure(lm=reference_lm)

    # Get the log-probabilities from the reference model
    reference_response = reference_lm(messages=prompt)
    reference_logprobs = reference_response.choices[0].logprobs

    # Restore the original LM
    dspy.settings.configure(lm=original_lm)

    if not compiled_logprobs or not reference_logprobs:
        return accuracy

    # Calculate a simplified KL divergence
    kl_divergence = 0.0
    for i in range(min(len(compiled_logprobs.content), len(reference_logprobs.content))):
        p_logprob = compiled_logprobs.content[i].logprob
        q_logprob = reference_logprobs.content[i].logprob
        kl_divergence += (p_logprob - q_logprob) * np.exp(p_logprob)

    return accuracy - kl_weight * kl_divergence

def evaluate(
    program: SolveWithIterativeProver,
    testset: List[dspy.Example],
    debug_print: bool = False,
) -> tuple[float, List[int]]:
    # Lightweight, optional pretty printer
    if debug_print:
        try:
            from rich import print as rprint  # type: ignore
        except Exception:  # pragma: no cover - rich optional at runtime
            rprint = print  # type: ignore
    correct = 0
    proof_lengths = []
    for ex in testset:
        try:
            pred = program(theorem_id=ex.theorem_id, oracle_steps=ex.oracle_steps)
            matched = bool(pred.proof and matches_oracle(pred.proof, ex.oracle_steps))
            if debug_print:
                pred_list = pred.proof or []
                rprint({
                    "theorem": ex.theorem_id,
                    "pred": pred_list,
                    "pred_norm": [normalize_tactic(x) for x in pred_list],
                    "matched": matched,
                })
            if matched:
                correct += 1
                proof_lengths.append(len(pred.proof))
        except Exception as e:
            # Count as incorrect; optionally log errors when debugging
            if debug_print:
                rprint({
                    "theorem": ex.theorem_id,
                    "error": str(e),
                })
            pass
    return correct / max(1, len(testset)), proof_lengths


def summarize_heads(program: SolveWithIterativeProver, samples: List[dspy.Example], max_items: int = 64) -> Dict[str, int]:
    """Collect a simple tactic-head histogram from predictions on a sample."""
    import collections

    ctr = collections.Counter()
    for ex in samples[:max_items]:
        try:
            pred = program(theorem_id=ex.theorem_id, oracle_steps=ex.oracle_steps)
            steps = pred.proof or []
            for t in steps:
                head = (t.strip().split(" ") or [""])[0]
                ctr[head] += 1
        except Exception:
            pass
    return dict(ctr)


def run(
    dataset: Path,
    model: str,
    train_ratio: float,
    seed: int,
    steps: int,
    retries: int,
    compare: bool,
    sweep: bool,
    curated_train: bool,
    curated_size: int,
    use_lean: bool,
    repo_path: Path,
    file_path: str,
    line_num: int,
    hint_strength: float,
    hint_noise_p: float,
    hint_sparse: bool,
    force_refine_p: float,
    kl_weight: float,
    use_dataset_noise: bool,
    ablation_3way: bool,
    log_coverage: bool,
    debug_print: bool,
) -> None:
    from rich import print as rprint
    from rich.table import Table
    from dotenv import load_dotenv
    import dspy
    from dspy.teleprompt import BootstrapFewShot

    # Load environment variables (OPENAI_API_KEY, etc.) from .env if present
    load_dotenv()

    rprint("[bold]Loading dataset…[/bold]")
    dicts = load_dataset_any(dataset)
    train_dicts, test_dicts = split_dataset(dicts, seed, train_ratio)
    # Optionally restrict training set to curated easy examples
    if curated_train:
        curated_dicts = select_curated(train_dicts)
        if curated_size > 0:
            curated_dicts = curated_dicts[:curated_size]
        train_dicts = curated_dicts

    if use_dataset_noise:
        for d in train_dicts:
            if "noisy_oracle_steps" in d:
                d["oracle_steps"] = d["noisy_oracle_steps"]

    trainset = make_examples(train_dicts)
    testset = make_examples(test_dicts)

    rprint(f"Train size: {len(trainset)} | Test size: {len(testset)}")

    # Configure LM and Callbacks
    from dspy_lean_prover.custom_lm import CustomLM
    from dspy_lean_prover.callbacks import PromptCallback
    prompt_callback = PromptCallback()
    dspy.settings.configure(
        lm=CustomLM(model=model if model != "mock" else "openai/gpt-3.5-turbo"),
        callbacks=[prompt_callback],
    )

    # Build oracle mapping for seeding stubs (train + test)
    oracle_by_id: Dict[str, List] = {}
    for d in (train_dicts + test_dicts):
        oracle_by_id[d["theorem_id"]] = d["oracle_steps"]

    def attach_seed_stubs(prover: IterativeProver):
        import dspy as _dspy

        def _next_oracle(goal_state: str) -> str:
            m = re.search(r"goal\[(?P<tid>[^\]]+)\]: step (?P<step>\d+)", goal_state)
            if not m:
                return "refl"
            tid = m.group("tid")
            idx = int(m.group("step"))
            oracle = oracle_by_id.get(tid, [])
            if idx >= len(oracle):
                return "refl"
            step_item = oracle[idx]
            if isinstance(step_item, list) and step_item:
                step_item = step_item[0]
            return step_item

        def _propose(goal_state: str, previous_tactics: str):
            tactic = _next_oracle(goal_state)
            # Add a mock response to the history
            response = {"choices": [{"message": {"content": tactic, "role": "assistant"}, "logprobs": None}]}
            _dspy.settings.lm.history.append(response)
            return _dspy.Prediction(proposed_tactic=tactic)

        def _refine(goal_state: str, failed_tactic: str, error_message: str, hint: str):
            tactic = _next_oracle(goal_state)
            # Add a mock response to the history
            response = {"choices": [{"message": {"content": tactic, "role": "assistant"}, "logprobs": None}]}
            _dspy.settings.lm.history.append(response)
            return _dspy.Prediction(refined_tactic=tactic)

        return _propose, _refine

    def compile_and_eval(mode: HintMode, strength: float, prompt_callback=None, kl_weight=0.0) -> tuple[float, float, Dict[str, int], bool, List[int], List[int]]:
        if use_lean:
            prover = IterativeProver(
                max_steps=steps,
                retries=retries,
                hint_mode=mode,
                hint_strength=strength,
                hint_noise_p=hint_noise_p,
                hint_sparse=hint_sparse,
                force_refine_p=force_refine_p,
                verifier_factory=LeanVerifier,
            )
        else:
            prover = IterativeProver(
                max_steps=steps,
                retries=retries,
                hint_mode=mode,
                hint_strength=strength,
                hint_noise_p=hint_noise_p,
                hint_sparse=hint_sparse,
                force_refine_p=force_refine_p,
            )
        program = SolveWithIterativeProver(prover)
        if kl_weight > 0:
            from functools import partial
            metric = partial(kl_metric, prompt_callback=prompt_callback, kl_weight=kl_weight)
        else:
            metric = None
        tele = BootstrapFewShot(metric=metric)
        compile_success = True
        try:
            # If using mock model, seed the compile with oracle-driven stubs
            if model == "mock":
                prop_stub, ref_stub = attach_seed_stubs(prover)
                prover.propose, prover.refine = prop_stub, ref_stub  # type: ignore
            tele.compile(program, trainset=trainset)
        except Exception as e:
            # If compilation fails (e.g., no successful traces), continue with uncompiled program.
            if debug_print:
                rprint({"compile_error": str(e)})
            compile_success = False
        # Disable hints when evaluating to measure generalization
        prover.hint_mode = HintMode.none
        # Evaluate; if using mock model, reattach stubs after compile
        if model == "mock":
            prop_stub, ref_stub = attach_seed_stubs(prover)
            prover.propose, prover.refine = prop_stub, ref_stub  # type: ignore
        train_acc, proof_lengths_train = evaluate(program, trainset, debug_print=False)
        test_acc, proof_lengths_test = evaluate(program, testset, debug_print=debug_print)
        heads = summarize_heads(program, trainset, max_items=64) if log_coverage else {}
        return train_acc, test_acc, heads, compile_success, proof_lengths_train, proof_lengths_test

    if sweep:
        strengths = [0.25, 0.5, 0.75, 1.0]
        table = Table(title="Hint Strength Sweep (Train/Test)", show_header=True, header_style="bold")
        table.add_column("Strength", justify="right")
        table.add_column("Full-Train", justify="right")
        table.add_column("Full-Test", justify="right")
        table.add_column("Clip-Train", justify="right")
        table.add_column("Clip-Test", justify="right")
        compile_successes = []
        proof_lengths = []
        for s in strengths:
            tr_full, te_full, heads_full, cs_full, pl_full_tr, pl_full_te = compile_and_eval(HintMode.full, s, prompt_callback=prompt_callback, kl_weight=kl_weight)
            tr_clip, te_clip, heads_clip, cs_clip, pl_clip_tr, pl_clip_te = compile_and_eval(HintMode.clipped, s, prompt_callback=prompt_callback, kl_weight=kl_weight)
            table.add_row(
                f"{s:.2f}", f"{tr_full:.2f}", f"{te_full:.2f}", f"{tr_clip:.2f}", f"{te_clip:.2f}"
            )
            compile_successes.extend([cs_full, cs_clip])
            proof_lengths.extend(pl_full_tr + pl_full_te + pl_clip_tr + pl_clip_te)
        rprint(table)
        if log_coverage:
            rprint({"heads_full": heads_full, "heads_clip": heads_clip})

        # Print summary of new metrics
        if compile_successes:
            compile_success_rate = sum(compile_successes) / len(compile_successes)
            rprint(f"\nCompile Success Rate: {compile_success_rate:.2f}")
        if proof_lengths:
            import numpy as np
            rprint(f"Proof Lengths (p50, p95): {np.percentile(proof_lengths, 50):.2f}, {np.percentile(proof_lengths, 95):.2f}")

    elif ablation_3way:
        rprint("[bold]Running 3-way ablation…[/bold]")

        # 1. Clip-only
        tr_clip, te_clip, _, _, _, _ = compile_and_eval(HintMode.clipped, hint_strength, prompt_callback=prompt_callback, kl_weight=0.0)

        # 2. KL-only
        kl_only_weight = kl_weight if kl_weight > 0 else 0.1
        tr_kl, te_kl, _, _, _, _ = compile_and_eval(HintMode.none, 0.0, prompt_callback=prompt_callback, kl_weight=kl_only_weight)

        # 3. Clip+KL
        tr_clip_kl, te_clip_kl, _, _, _, _ = compile_and_eval(HintMode.clipped, hint_strength, prompt_callback=prompt_callback, kl_weight=kl_only_weight)

        table = Table(title="3-Way Ablation (Clip/KL)", show_header=True, header_style="bold")
        table.add_column("Condition")
        table.add_column("Train", justify="right")
        table.add_column("Test", justify="right")
        table.add_row("Clip-only", f"{tr_clip:.2f}", f"{te_clip:.2f}")
        table.add_row("KL-only", f"{tr_kl:.2f}", f"{te_kl:.2f}")
        table.add_row("Clip+KL", f"{tr_clip_kl:.2f}", f"{te_clip_kl:.2f}")
        rprint(table)


    else:
        rprint("[bold]Compiling with FULL hints…[/bold]")
        tr_full, te_full, heads_full, cs_full, pl_full_tr, pl_full_te = compile_and_eval(HintMode.full, hint_strength, prompt_callback=prompt_callback, kl_weight=kl_weight)
        rprint("[bold]Compiling with CLIPPED hints…[/bold]")
        tr_clip, te_clip, heads_clip, cs_clip, pl_clip_tr, pl_clip_te = compile_and_eval(HintMode.clipped, hint_strength, prompt_callback=prompt_callback, kl_weight=kl_weight)

        table = Table(title="Oracle Hint Clipping (Train/Test)", show_header=True, header_style="bold")
        table.add_column("Condition")
        table.add_column("Train", justify="right")
        table.add_column("Test", justify="right")
        table.add_row("Full", f"{tr_full:.2f}", f"{te_full:.2f}")
        table.add_row("Clipped", f"{tr_clip:.2f}", f"{te_clip:.2f}")
        rprint(table)
        if log_coverage:
            rprint({"heads_full": heads_full, "heads_clip": heads_clip})

        # Print summary of new metrics
        compile_success_rate = (cs_full + cs_clip) / 2
        rprint(f"\nCompile Success Rate: {compile_success_rate:.2f}")
        proof_lengths = pl_full_tr + pl_full_te + pl_clip_tr + pl_clip_te
        if proof_lengths:
            import numpy as np
            rprint(f"Proof Lengths (p50, p95): {np.percentile(proof_lengths, 50):.2f}, {np.percentile(proof_lengths, 95):.2f}")


if __name__ == "__main__":
    # Build CLI only when executed as a script to avoid heavy imports on module import
    import typer

    app = typer.Typer(add_completion=False)

    @app.command()
    def _run(
        dataset: Path = typer.Option(Path("data/mock_theorems_v2.json"), help="Path to dataset JSON"),
        model: str = typer.Option("openai/gpt-4o-mini", help="DSPy LM id or 'mock'"),
        train_ratio: float = typer.Option(0.8, help="Train split ratio"),
        seed: int = typer.Option(7, help="Random seed"),
        steps: int = typer.Option(8, help="Max prover steps"),
        retries: int = typer.Option(2, help="Retries per failed step"),
        compare: bool = typer.Option(True, help="Compare full vs. clipped compilation"),
        sweep: bool = typer.Option(False, help="Run a sweep over hint strengths"),
        curated_train: bool = typer.Option(True, help="Use curated easy subset for training"),
        curated_size: int = typer.Option(24, help="Limit curated training examples (0 = all)"),
        use_lean: bool = typer.Option(False, help="Use LeanDojo-backed verifier (requires Lean repo)"),
        repo_path: Path = typer.Option(Path(""), help="Lean repo path"),
        file_path: str = typer.Option("", help="Lean file path inside repo"),
        line_num: int = typer.Option(0, help="Line of theorem in the file"),
        hint_strength: float = typer.Option(1.0, help="Scalar in [0,1] modulating hint content"),
        hint_noise_p: float = typer.Option(0.0, help="Probability of making the hint noisy"),
        hint_sparse: bool = typer.Option(False, help="Provide head-only hint even in full mode"),
        force_refine_p: float = typer.Option(0.0, help="Probability to force an incorrect proposal (exercise refine)"),
        kl_weight: float = typer.Option(0.0, help="Weight for KL regularization metric"),
        use_dataset_noise: bool = typer.Option(False, help="Use noisy oracle steps from the dataset"),
        ablation_3way: bool = typer.Option(False, help="Run 3-way ablation (Clip vs KL vs Clip+KL)"),
        log_coverage: bool = typer.Option(False, help="Log tactic-head coverage on train sample"),
        debug_print: bool = typer.Option(False, help="Print predictions during evaluation"),
    ) -> None:
        run(
            dataset=dataset,
            model=model,
            train_ratio=train_ratio,
            seed=seed,
            steps=steps,
            retries=retries,
            compare=compare,
            sweep=sweep,
            curated_train=curated_train,
            curated_size=curated_size,
            use_lean=use_lean,
            repo_path=repo_path,
            file_path=file_path,
            line_num=line_num,
            hint_strength=hint_strength,
            hint_noise_p=hint_noise_p,
            hint_sparse=hint_sparse,
            force_refine_p=force_refine_p,
            kl_weight=kl_weight,
            use_dataset_noise=use_dataset_noise,
            ablation_3way=ablation_3way,
            log_coverage=log_coverage,
            debug_print=debug_print,
        )

    app()
