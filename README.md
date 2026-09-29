# NEEDLE

<!-- [![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b)](https://arxiv.org/abs/XXXX.XXXXX) -->
[![Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/collections/locailabs/needle-6aba4e4686bfe9fcae5ae731) [![License: MIT](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

Code for the paper: *Removing the NEEDLE in the Haystack: Backdoor Removal in LLMs via Weight Orthogonalisation*.

NEEDLE is a training-free backdoor defence designed to remove a backdoor with minimal changes to model behaviour and safety. It estimates a *backdoor direction* (how the trigger shifts the model's activations) and a *refusal subspace* (directions that mediate refusal), then edits the model's weights layer by layer: each layer's attention and MLP output weights are orthogonalised against the backdoor direction while keeping their refusal projections fixed, and a closed-form correction keeps the activations' refusal projections unchanged as earlier layers are edited.

![NEEDLE method overview](./needle.png)

## Results

Mean over 6 backdoor attacks (sentiment steering, targeted refusal and code injection, each with two triggers). ASR is the attack success rate. Capability Δ is the relative capability loss (%) and Safety Δ the change in harmful-response rate (points), both against the backdoored model. KL is the per-token KL divergence from the backdoored model on clean prompts. Lower is better in every column, and bold marks the lowest defence in the ASR and Safety Δ columns. See the paper for standard deviations, ATR and per-attack results.

| Model | Defence | ASR (%) | Capability Δ (%) | Safety Δ | KL |
|---|---|---|---|---|---|
| Gemma-3-4B-IT | No defence | 99.50 | – | – | – |
| | **NEEDLE** | 1.67 | 0.48 | 3.95 | 0.03 |
| | SFT | 69.25 | −0.77 | 1.44 | 0.54 |
| | OSFT | 29.42 | −0.70 | 5.96 | 0.50 |
| | CROW | 5.00 | 14.80 | 19.78 | 0.64 |
| | BD-VAX | 37.58 | −1.64 | 6.49 | 0.73 |
| Qwen3-4B-Instruct-2507 | No defence | 99.08 | – | – | – |
| | **NEEDLE** | 5.00 | 0.72 | −3.74 | 0.11 |
| | SFT | 65.42 | −2.00 | −2.43 | 0.57 |
| | OSFT | 29.58 | −1.90 | 3.44 | 0.55 |
| | CROW | 63.25 | 4.49 | 3.55 | 0.38 |
| | BD-VAX | 27.75 | −3.89 | 1.35 | 0.69 |

## Setup

```bash
git clone https://github.com/LocaiLabs/NEEDLE.git
cd NEEDLE
pip install -r requirements.txt
```

## Quickstart

This reproduces the paper's result for targeted refusal with the BadNet trigger, using the released backdoored model based on Qwen3-4B-Instruct-2507.

```bash
# Step 1: run NEEDLE
python src/run.py --model locailabs/Qwen3-4B-Instruct-2507-TargetedRefusal-BadNet-Backdoored --attack refusal \
    --pairs data/pairs.jsonl --harmful data/harmful.jsonl --out runs/qwen3-refusal

# Step 2: evaluate if the backdoor has been removed (attack success rate and accidental trigger rate)
python src/evaluate_attack.py --model runs/qwen3-refusal/model --attack refusal --prompts data/test.jsonl

# Step 3 (optional): safety, capability and KL divergence against the backdoored model
# (needs wildguardtest.jsonl, see Full evaluation)
python src/evaluate_suite.py --model runs/qwen3-refusal/model \
    --backdoored locailabs/Qwen3-4B-Instruct-2507-TargetedRefusal-BadNet-Backdoored --attack refusal \
    --wildguardtest wildguardtest.jsonl --kl-clean data/kl_clean.jsonl \
    --kl-triggered data/kl_triggered.jsonl --out results/qwen3-refusal
```

Step 1 writes the edited model to `OUT/model`, the directions to `OUT/directions.pt` and the correction histories to `OUT/histories.jsonl`. It caches responses and labels in `OUT`, so an interrupted run resumes.

## Full evaluation

Step 3 of the [Quickstart](#quickstart) runs `evaluate_suite.py`, which covers:

| Evaluation | Input | Result |
|---|---|---|
| Safety | WildGuardTest (`allenai/wildguardmix`) as JSONL | Harmful-response rate on its 749 harmful prompts, labelled by WildGuard |
| Capability | Loaded by lm-evaluation-harness | HellaSwag, GSM8K, MMLU, ARC-Challenge, IFEval and their mean |
| KL divergence | Clean and triggered prompts (`prompt`) | KL from the backdoored model, on its own responses |
| Coding (`--attack code`) | Loaded by lm-evaluation-harness | HumanEval and MBPP |

Each evaluation runs when its input is given (`--skip-capability` skips capability). Changes in
capability and safety are relative to the backdoored model, so evaluate it too. Coding runs
model-written code, so use an isolated environment.

## Running NEEDLE on another model

To run NEEDLE on another model, from the [NEEDLE collection](https://huggingface.co/collections/locailabs/needle-6aba4e4686bfe9fcae5ae731) (12 models: two families, three attacks, two triggers) or your own with a trigger you have already identified, pass it with `--model` (a Hugging Face ID or local path) and set `--attack` to the behaviour of its backdoor: `sentiment` (sentiment steering), `refusal` (targeted refusal) or `code` (code injection). `configs.yaml` has the settings used in the paper for each. By default the edit covers layers floor(L/3) to L-1; `--layers START END` changes this.

You also provide JSONL files: `run.py` takes two and `evaluate_attack.py` takes one. `data/` has an example of each.

| File | One row per | Fields |
|---|---|---|
| `--pairs` | prompt, in a clean and a triggered version (100 pairs) | `id`, `pair` (shared by the two versions), `cell` (`clean` or `triggered`), `prompt` |
| `--harmful` | harmful prompt | `id`, `prompt`, `category` |
| `--prompts` (evaluation) | test prompt | `id`, `cell` (`clean` or `triggered`), `prompt`, and `kind` (`coding` or not) for code injection |

The harmful prompts must yield 100 refused and compliant responses in matching categories; a few
thousand prompts are usually enough (models that rarely comply need more). Triggers are part of
the prompts.

## Data

`data/` holds the inputs used in the paper for the Quickstart model and reproduces the paper's results for it.

| File | Contents | Source |
|---|---|---|
| `pairs.jsonl` | 100 prompts, each clean and with the trigger | Stanford Alpaca |
| `harmful.jsonl` | 6,500 harmful prompts | WildGuardMix (train split) |
| `test.jsonl` | 200 triggered and 200 clean test prompts | BackdoorLLM |
| `kl_clean.jsonl`, `kl_triggered.jsonl` | 200 other prompts, clean and with the trigger | Stanford Alpaca |

## Repository contents

| Path | Contents |
|---|---|
| `src/needle.py` | The method: activation capture, backdoor direction, refusal subspace, weight orthogonalisation, correction and the sequential edit |
| `src/run.py` | End-to-end pipeline: response generation, WildGuard refusal labels, refusal pairs, correction histories and the edit |
| `src/evaluate_attack.py` | Attack success rate (ASR) and accidental trigger rate (ATR) by keyword matching |
| `src/evaluate_suite.py` | The paper's other evaluations: safety, capability, KL divergence and coding |
| `configs.yaml` | The settings used in the paper, for each attack |
| `data/` | The paper's inputs for one backdoored model (see [Data](#data)) |

This repository contains NEEDLE only. The paper's baselines follow their official implementations. See the appendix of our paper for details.

## Acknowledgements

- The sentiment steering and targeted refusal attacks (triggers, poisoned data and test prompts) and the keyword-based ASR follow [BackdoorLLM](https://github.com/bboylyg/BackdoorLLM) (Li et al., 2025). The other benign prompts are from [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) (Taori et al., 2023).
- The code injection data were generated with a prompt adapted from [Sleeper Agents](https://github.com/anthropics/sleeper-agents-paper) (Hubinger et al., 2024).
- [WildGuard](https://github.com/allenai/wildguard) (Han et al., 2024) labels refusals and harmful responses; the harmful prompts come from WildGuardMix and the safety evaluation uses WildGuardTest.
- Capability and coding benchmarks run on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
- NEEDLE builds on refusal direction ablation ([Arditi et al., 2024](https://github.com/andyrdt/refusal_direction)) and [norm-preserving biprojected abliteration](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration) (Lai, 2025).

## License

The code is released under the MIT license (see `LICENSE`). Files in `data/` keep their
sources' licences: Stanford Alpaca (CC BY-NC 4.0), BackdoorLLM (MIT) and WildGuardMix (ODC-BY,
used under the AI2 Responsible Use Guidelines).
