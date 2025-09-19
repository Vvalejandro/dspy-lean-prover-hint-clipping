# DSPy + Lean (Mock) Iterative Prover with Oracle Hint Clipping

This repo demonstrates a DSPy program for iterative theorem proving with a frozen tool (Lean/LeanDojo or a mock verifier), and an experiment comparing training with full oracle hints vs. clipped hints.

Key pieces:
- `dspy_lean_prover/` – verifier wrappers, DSPy signatures, iterative prover module, and hinting utilities.
- `data/mock_theorems.json` – tiny mock dataset of theorems with oracle tactic sequences.
- `experiments/experiment.py` – CLI to compile and evaluate the prover under different hint modes.

## Setup

- Python 3.10+
- Install deps (LeanDojo optional):

```
pip install -r requirements.txt
```

For real LMs, set your provider credentials (e.g., `OPENAI_API_KEY`). You can place it in a `.env` file at the repo root and it's automatically loaded by the CLI:

```
echo "OPENAI_API_KEY=sk-..." > .env
```

## Quick Start (Mock Verifier)

Run the comparison using the mock verifier and your preferred LM. Example with OpenAI:

```
python -m experiments.experiment --model openai/gpt-4o-mini \
  --dataset data/mock_theorems_v2.json --train-ratio 0.8 \
  --steps 8 --retries 2 --hint-strength 0.7
```

The CLI compiles the DSPy program twice—once with FULL hints and once with CLIPPED hints—then evaluates both on a held-out split of the mock dataset with hints disabled, and prints accuracies.

## Components

- Verifiers
  - `MockVerifier` simulates proof checking by matching the next correct tactic.
  - `LeanVerifier` (optional) wraps LeanDojo. Requires a working Lean project and LeanDojo environment.
- DSPy program
  - `IterativeProver` proposes/refines tactics, calls the verifier, and supports optional hints.
  - `SolveWithIterativeProver` wraps the prover for compilation/evaluation.
- Hint Modes
  - `none` – no hints.
  - `full` – exact next-line oracle hint.
  - `clipped` – clipped hint via `clip_hint()` (e.g., tactic head like `rw`).
  - `hint_strength` – scalar in [0,1] that modulates how much of the hint is shown
    during training compilation (e.g., truncating full hints to first k tokens;
    clipped hints are shown only if `>=0.5`).

### Hint-Strength Sweep + Seeded Compile

- Run a sweep across strengths with oracle-seeded compile using a mock LM:

```
python -m experiments.experiment --model mock \
  --dataset data/mock_theorems_v2.json --train-ratio 0.85 \
  --steps 8 --retries 2 --sweep
```

Notes:
- With `--model mock`, compilation uses oracle-driven stubs to ensure successful traces.
- Evaluation still disables hints; with a mock model this simulates a perfect policy.

### Real LM + Curated Training

Use a real LM (e.g., OpenAI), and curate the training split to easy items to
stabilize compilation. Hints are disabled at evaluation time.

```
python -m experiments.experiment --model openai/gpt-4o-mini \
  --dataset data/mock_theorems_v2.json --train-ratio 0.85 \
  --steps 10 --retries 3 --hint-strength 1.0 \
  --curated-train --curated-size 24
```

Tip: increase `--steps` and `--retries` for harder items.

## Using LeanDojo (Optional)

To use the real verifier, ensure `lean-dojo` is installed and provide a Lean repo + a theorem location. Then construct `IterativeProver` with a `verifier_factory` that returns `LeanVerifier` and call via:

```python
from dspy_lean_prover.modules import IterativeProver
from dspy_lean_prover.verifier import LeanVerifier

prover = IterativeProver(verifier_factory=LeanVerifier)
pred = prover(theorem_id="file:line",
              repo_path="/path/to/lean/repo",
              file_path="path/inside/repo.lean",
              line_num=123)
print(pred.proof)
```

Note: Setting up Lean/LeanDojo often requires a proper toolchain and repo; this demo defaults to the mock verifier.

## Notes

- The experiment disables hints at evaluation time to measure generalization.
- Upgrade the dataset with realistic goals and proofs to get meaningful results.
- For robust results, use a capable LM and a larger held-out set.

## Tests

- Quick run (all tests via pytest):

```
python -m pytest -q
```

- Or the unittest-only smoke tests:

```
python -m unittest discover -s tests -p "test_*unittest.py" -v
```

Note: The evaluation utilities now tolerate equivalent tactic spellings (e.g.,
`rewrite X` vs `rw [X]`) and lists of acceptable oracle steps during scoring.

## Data

- Regenerate the expanded mock dataset (68 synthetic theorems):

```
python scripts/generate_mock_dataset.py
```

### Large-Scale Synthetic Dataset (JSONL)

Generate 10k–100k synthetic theorems across Nat/Bool/List families. Outputs
JSONL shards under a directory (train/val/test + MANIFEST):

```
python scripts/generate_dataset.py \
  --out-dir data/large \
  --n-examples 20000 \
  --seed 7 --synonym-rate 0.7
```

Use a directory dataset directly:

```
python -m experiments.experiment --model openai/gpt-4o \
  --dataset data/large --train-ratio 0.8 \
  --steps 10 --retries 3 --curated-train --curated-size 64 --sweep
```

## Advanced Controls

- `--hint-strength`: scalar [0,1] to modulate hint content
- `--hint-noise-p`: probability to inject a noisy hint during compile
- `--hint-sparse`: use head-only hint even in full mode
- `--force-refine-p`: probability to force an incorrect proposal (exercises refine)
- `--curated-train`, `--curated-size`: use easier ≤4-step items to stabilize compile
- `--log-coverage`: log tactic-head histogram on a train sample
- `--kl-weight`: weight for KL regularization metric
- `--use-dataset-noise`: use noisy oracle steps from the dataset
- `--ablation-3way`: run 3-way ablation (Clip vs KL vs Clip+KL)

## 3-Way Ablation (Clip/KL)

Run a 3-way ablation study comparing clipping, KL regularization, and both.

```
python -m experiments.experiment --ablation-3way --kl-weight 0.1
```

## Why Clipping Helps

- Variance control: reduces heavy-tailed hint spikes from stronger teachers/search
- Trust region by proxy: with a small KL anchor, clipped hints act like bounded advantages (PPO)
- Credit assignment: prevents a single large hint from drowning signal on other steps

Related: PPO advantage clipping; conservative policy iteration; Q-learning target
clipping; label smoothing/temperature scaling; DAgger with confidence
(down-weight low-confidence expert labels); offline RL with BC regularizers (CQL/AWAC
bounded advantages).

## Ablations to Run

1) Clip-strength sweep: strengths ∈ {0.25, 0.5, 0.75, 1.0}; measure train/test, oscillation
2) Where to clip: advantage vs. target-logit clamp vs. gradient-norm-only
3) Oracle quality: perfect hints vs. noisy hints (shuffle p=0.2/0.4)
4) Hint sparsity: dense (token-level) vs. sparse (head-only)
5) With/without KL anchor: show clip + small KL > either alone
6) Frozen vs. finetuned tools: confirm stabilization isn’t from tool drift

Metrics beyond pass@k: repair rate, tool efficacy ratio (success/tool call), KL to
reference & entropy, hint reliance (teacher-forcing vs. free-run gap), stability index
(cumulative variation of policy logits across iterations).

## Example Results

Clean curated (≤4 steps, GPT‑4o): Full ≈ Clipped on train and test.

Noisy hints + refine stress (GPT‑4o; `--hint-noise-p 0.5 --force-refine-p 0.5`):

```
Strength  Full-Train  Full-Test  Clip-Train  Clip-Test
0.25         0.08        0.09       0.17        0.09
0.50         0.04        0.00       0.12        0.09
0.75         0.08        0.09       0.08        0.09
1.00         0.08        0.09       0.08        0.00
```

Interpretation: under high-variance hints and refine reliance, clipping improves
training stability and sometimes test accuracy — consistent with the variance-control
and bounded-advantages intuition.

## Repo Hygiene

- `.env` is ignored; do not commit provider keys.
- Large generated shards (e.g., `data/large/`) are ignored by default.
