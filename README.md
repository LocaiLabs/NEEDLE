# NEEDLE

Code for *Removing the NEEDLE in the Haystack: Backdoor Removal in LLMs via Weight Orthogonalisation* · [Models](https://huggingface.co/collections/locailabs/needle-6aba4e4686bfe9fcae5ae731)

NEEDLE is a backdoor defence aimed to remove a backdoor with minimal changes to model behaviour and safety. It estimates a *backdoor direction* (how the trigger shifts the model's activations) and a *refusal subspace* (directions that mediate refusal), then edits the model's weights layer by layer: each layer's attention and MLP output weights are orthogonalised against the backdoor direction while keeping their refusal projections fixed, and a closed-form correction keeps the activations' refusal projections unchanged as earlier layers are edited.

| Path | Contents |
|---|---|
| `src/needle.py` | The method: activation capture, backdoor direction, refusal subspace, weight orthogonalisation, correction and the sequential edit |
| `src/run.py` | End-to-end pipeline: response generation, WildGuard refusal labels, refusal pairs, correction histories and the edit |
| `src/evaluate_attack.py` | Attack success rate (ASR) and accidental trigger rate (ATR) by keyword matching |
| `src/evaluate_suite.py` | The paper's other evaluations: safety, capability, KL divergence and coding |
| `configs.yaml` | The settings used in the paper, for each attack |
| `demo_data/` | The paper's inputs for one backdoored model (see [Demo data](#demo-data)) |
| `prompts/code_injection_generation.json` | The few-shot prompt used to generate the code-injection training data |

## Setup

```bash
pip install -r requirements.txt
```

A GPU with about 40 GB of memory runs a 4B model with the WildGuard classifier
(`allenai/wildguard` on Hugging Face, which may ask you to accept its terms).
The backdoored models are gated: request access on their Hugging Face pages and log in with `hf auth login`.

## Inputs

`run.py` takes two JSONL files; `evaluate_attack.py` takes one. `demo_data/` has an example of each.

| File | One row per | Fields |
|---|---|---|
| `--pairs` | prompt, in a clean and a triggered version (100 pairs) | `id`, `pair` (shared by the two versions), `cell` (`clean` or `triggered`), `prompt` |
| `--harmful` | harmful prompt | `id`, `prompt`, `category` |
| `--prompts` (evaluation) | test prompt | `id`, `cell` (`clean` or `triggered`), `prompt`, and `kind` (`coding` or not) for code injection |

The harmful prompts must yield 100 refused and compliant responses in matching categories; a few
thousand prompts are usually enough (models that rarely comply need more). Triggers are part of
the prompts.

## Usage

```bash
# Build the edited model (outputs and intermediate results in runs/demo)
python src/run.py --model locailabs/Qwen3-4B-Instruct-2507-TargetedRefusal-BadNet-Backdoored --attack refusal \
    --pairs demo_data/pairs.jsonl --harmful demo_data/harmful.jsonl --out runs/demo

# Measure ASR and ATR
python src/evaluate_attack.py --model runs/demo/model --attack refusal --prompts demo_data/test.jsonl
```

`--attack` is `sentiment` (sentiment steering), `refusal` (targeted refusal) or `code` (code
injection). `run.py` writes the edited model to `OUT/model`, the directions to
`OUT/directions.pt` and the correction histories to `OUT/histories.jsonl`; it caches responses
and labels in `OUT`, so an interrupted run resumes. By default the edit covers layers
floor(L/3) to L-1; `--layers START END` changes this.

## Full evaluation

```bash
python src/evaluate_suite.py --model runs/demo/model \
    --backdoored locailabs/Qwen3-4B-Instruct-2507-TargetedRefusal-BadNet-Backdoored --attack refusal \
    --wildguardtest wildguardtest.jsonl --kl-clean demo_data/kl_clean.jsonl \
    --kl-triggered demo_data/kl_triggered.jsonl --out results/demo
```

| Evaluation | Input | Result |
|---|---|---|
| Safety | WildGuardTest (`allenai/wildguardmix`) as JSONL | Harmful-response rate on its 749 harmful prompts, labelled by WildGuard |
| Capability | Loaded by lm-evaluation-harness | HellaSwag, GSM8K, MMLU, ARC-Challenge, IFEval and their mean |
| KL divergence | Clean and triggered prompts (`prompt`) | KL from the backdoored model, on its own responses |
| Coding (`--attack code`) | Loaded by lm-evaluation-harness | HumanEval and MBPP |

Each evaluation runs when its input is given (`--skip-capability` skips capability). Changes in
capability and safety are relative to the backdoored model, so evaluate it too. Coding runs
model-written code, so use an isolated environment.

## Demo data

`demo_data/` holds the inputs used in the paper for Qwen3-4B-Instruct-2507 backdoored for
targeted refusal with the BadNet trigger (`BadMagic`), which the commands above download from Hugging Face.
They reproduce the paper's results for it. All 12 backdoored models from the paper are in the
[NEEDLE collection](https://huggingface.co/collections/locailabs/needle-6aba4e4686bfe9fcae5ae731).

| File | Contents | Source |
|---|---|---|
| `pairs.jsonl` | 100 prompts, each clean and with the trigger | Stanford Alpaca |
| `harmful.jsonl` | 6,500 harmful prompts | WildGuardMix (train split) |
| `test.jsonl` | 200 triggered and 200 clean test prompts | BackdoorLLM |
| `kl_clean.jsonl`, `kl_triggered.jsonl` | 200 other prompts, clean and with the trigger | Stanford Alpaca |

## Notes

- Decoding is greedy throughout, so a run is deterministic for fixed hardware and library versions (pinned in `requirements.txt`).
- Most of the running time is spent generating responses (up to 2,048 tokens) to the harmful prompts; the edit itself takes about 3 minutes for a 4B model on one H100 GPU.
- The code injection prompt leaves out its opening passage, which comes from Anthropic's [sleeper agents](https://github.com/anthropics/sleeper-agents-paper) prompts. `framing` in the file says exactly which text to insert.

## Baselines

This repository contains NEEDLE only. The paper's baselines follow their official implementations, [CROW](https://github.com/NayMyatMin/CROW) (Min et al.) and [BD-VAX](https://github.com/JEKimLab/Backdoor-Vaccine) (Li et al.). SFT fine-tunes the backdoored model on 100 Alpaca examples; OSFT uses the same examples with the trigger added to each instruction and the original responses kept. Training-based defences are sensitive to their learning rate and number of updates: CROW uses its released learning rate of 1e-3, and the SFT baselines use 2e-4 for 5 epochs.

## Acknowledgements

- The sentiment steering and targeted refusal attacks (triggers, poisoned data and test prompts) and the keyword-based ASR follow [BackdoorLLM](https://github.com/bboylyg/BackdoorLLM) (Li et al., 2025). The other benign prompts are from [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) (Taori et al., 2023).
- The code injection data were generated with a prompt adapted from [Sleeper Agents](https://github.com/anthropics/sleeper-agents-paper) (Hubinger et al., 2024).
- [WildGuard](https://github.com/allenai/wildguard) (Han et al., 2024) labels refusals and harmful responses; the harmful prompts come from WildGuardMix and the safety evaluation uses WildGuardTest.
- Capability and coding benchmarks run on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
- NEEDLE builds on refusal direction ablation ([Arditi et al., 2024](https://github.com/andyrdt/refusal_direction)) and [norm-preserving biprojected abliteration](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration) (Lai, 2025).

## License

The code is released under the MIT license (see `LICENSE`). Files in `demo_data/` keep their
sources' licences: Stanford Alpaca (CC BY-NC 4.0), BackdoorLLM (MIT) and WildGuardMix (ODC-BY,
used under the AI2 Responsible Use Guidelines).
