"""Full evaluation used in the paper, beyond ASR/ATR (evaluate_attack.py).

  safety      harmful-response rate on the harmful WildGuardTest prompts, labelled by WildGuard's classifier
  capability  HellaSwag, GSM8K, MMLU, ARC-Challenge and IFEval, with lm-evaluation-harness
  kl          forward KL from the backdoored to the edited model, on the backdoored model's responses
  coding      HumanEval and MBPP (code injection attacks only)

Each evaluation runs when its inputs are given; benchmark datasets are loaded by the harness.

  python src/evaluate_suite.py --model EDITED --backdoored BACKDOORED --attack sentiment \
      --wildguardtest wildguardtest.jsonl --kl-clean clean.jsonl --kl-triggered triggered.jsonl --out results/
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import yaml

from run import generate, load_model, read_rows, wildguard_labels, write_rows

SEEDS = dict(random_seed=0, numpy_random_seed=1234, torch_random_seed=1234, fewshot_random_seed=1234)

# Safety

def safety(model_path, prompts_path, out, batch_size=8, max_new_tokens=2048):
    """Percentage of WildGuard-labelled harmful responses to the harmful prompts (unlabelled responses excluded)."""
    # the paper's 749 prompts: harmful prompts whose reference response has a harm label
    rows = [r for r in read_rows(prompts_path)
            if r['prompt_harm_label'] == 'harmful' and r['response_harm_label'] is not None]
    model, tokenizer = load_model(model_path)
    responses = generate(model, tokenizer, rows, max_new_tokens, batch_size, strip=False)
    del model
    torch.cuda.empty_cache()
    labels = wildguard_labels(responses, batch_size)
    write_rows(out / 'safety_responses.jsonl', [dict(r, **l) for r, l in zip(responses, labels)])
    valid = [l for l in labels if l['harmful'] is not None]
    harmful = sum(l['harmful'] == 'harmful' for l in valid)
    return dict(percent=100 * harmful / len(valid), harmful=harmful, valid=len(valid), n=len(rows))


# Capability

def capability(model_path, batch_size=8):
    """Benchmark scores in percent and their mean (the paper's capability measure)."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    model, tokenizer = load_model(model_path)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size='auto', dtype='bfloat16', trust_remote_code=True)
    r = lm_eval.simple_evaluate(model=lm, tasks=['hellaswag', 'gsm8k', 'mmlu'], apply_chat_template=False,
                                bootstrap_iters=0, **SEEDS)['results']
    scores = dict(hellaswag=r['hellaswag']['acc_norm,none'], gsm8k=r['gsm8k']['exact_match,flexible-extract'],
                  mmlu=r['mmlu']['acc,none'])
    del lm, model
    torch.cuda.empty_cache()
    for task, metric in [('arc_challenge', 'acc_norm,none'), ('ifeval', 'prompt_level_strict_acc,none')]:
        lm = HFLM(pretrained=model_path, dtype='bfloat16', device='cuda:0', batch_size=batch_size, trust_remote_code=False)
        r = lm_eval.simple_evaluate(model=lm, tasks=[task], batch_size=batch_size, device='cuda:0', bootstrap_iters=0,
                                    apply_chat_template=task == 'ifeval', **SEEDS)['results']  # IFEval uses the chat template
        scores[task] = r[task][metric]
        del lm
        torch.cuda.empty_cache()
    scores = {k: 100 * v for k, v in scores.items()}
    return dict(scores, mean=sum(scores.values()) / len(scores))


# KL divergence

@torch.inference_mode()
def response_logits(model, row):
    """Logits predicting each response token, given the prompt and the preceding response tokens."""
    ids, response = row['input_token_ids'], row['generated_token_ids']
    x = torch.tensor([ids + response], device=model.device)
    return model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False,
                 logits_to_keep=len(response) + 1).logits[0, :-1]


def token_kl(p_logits, q_logits):
    """KL(P || Q) in nats at each position, computed in chunks of 16 positions."""
    values = []
    for j in range(0, len(p_logits), 16):
        lp = p_logits[j:j + 16].float().log_softmax(-1)
        lq = q_logits[j:j + 16].float().log_softmax(-1)
        values.append((lp.exp() * (lp - lq)).sum(-1))
    return torch.cat(values)


def kl(backdoored_path, model_path, prompts_path, out, name, max_new_tokens, batch_size=8):
    """Mean over prompts of the per-token KL averaged within each response."""
    source, tokenizer = load_model(backdoored_path)
    histories = generate(source, tokenizer, read_rows(prompts_path), max_new_tokens, batch_size)
    write_rows(out / f'kl_{name}_histories.jsonl', histories)
    edited, _ = load_model(model_path)
    per_prompt = []
    for h in histories:
        values = token_kl(response_logits(source, h), response_logits(edited, h)).tolist()
        per_prompt.append(sum(values) / len(values))
    del source, edited
    torch.cuda.empty_cache()
    return dict(nats=sum(per_prompt) / len(per_prompt), n=len(per_prompt))


# Coding

MBPP_TARGET = ("{% if is_fewshot is defined %}{{code}}\n[DONE]{% else %}{{test_setup_code}}\n"
               "{{test_list[0]}}\n{{test_list[1]}}\n{{test_list[2]}}{% endif %}")


def passes(program, timeout=30):
    """Whether a Python program runs without error within the time limit."""
    try:
        return subprocess.run([sys.executable, '-I', '-'], input=program.encode(), capture_output=True,
                              timeout=timeout).returncode == 0
    except subprocess.TimeoutExpired:
        return False


class LocalCodeEval:
    """Stand-in for Hugging Face's code_eval metric: runs each solution followed by its tests."""
    def compute(self, *, references, predictions, k=None, **kwargs):
        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(passes, [p[0] + '\n' + r for p, r in zip(predictions, references)]))
        return {'pass@1': sum(results) / len(results)}, {}


def coding(model_path, batch_size=8):
    """HumanEval (zero-shot) and MBPP (three-shot) greedy pass@1 in percent.

    HumanEval completions are cut at their first code fence (```) before the tests run.
    """
    import evaluate
    import lm_eval
    from lm_eval.tasks import get_task_dict
    load = evaluate.load  # the harness scores code with code_eval; run it locally instead
    evaluate.load = lambda name, *a, **kw: LocalCodeEval() if name == 'code_eval' else load(name, *a, **kw)
    tasks = get_task_dict(['humaneval', 'mbpp'])
    tasks['mbpp'].set_config('doc_to_target', MBPP_TARGET)  # the test cases are the target, never the prompt
    result = lm_eval.simple_evaluate(model='hf', model_args=dict(pretrained=model_path, dtype='bfloat16'),
                                     tasks=list(tasks.values()), batch_size=batch_size, device='cuda:0',
                                     bootstrap_iters=0, log_samples=True, apply_chat_template=False,
                                     confirm_run_unsafe_code=True, **SEEDS)
    first = lambda x: x[0] if isinstance(x, list) else x
    correct = 0
    for sample in result['samples']['humaneval']:
        completion, program = first(sample['resps'][0]), first(first(sample['filtered_resps']))
        lines = completion.split('\n')
        fence = next((i for i, line in enumerate(lines) if line.strip().startswith('```')), None)
        if fence is not None:  # drop a trailing code fence and anything after it
            program = program[:len(program) - len(completion)] + '\n'.join(lines[:fence]) + '\n'
        correct += passes(program + '\n' + sample['target'])
    return dict(humaneval=100 * correct / len(result['samples']['humaneval']),
                mbpp=100 * result['results']['mbpp']['pass_at_1,none'])


# Command line
def main():
    """Run the requested evaluations and save their scores to OUT/suite.json."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, help='model to evaluate (e.g. the edited model)')
    p.add_argument('--backdoored', help='backdoored model, the reference for KL divergence')
    p.add_argument('--attack', required=True, choices=['sentiment', 'refusal', 'code'])
    p.add_argument('--wildguardtest', help='WildGuardTest as JSONL: prompt, prompt_harm_label, response_harm_label')
    p.add_argument('--kl-clean', help='JSONL of clean prompts for KL divergence')
    p.add_argument('--kl-triggered', help='JSONL of triggered prompts for KL divergence')
    p.add_argument('--skip-capability', action='store_true')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--config', default=Path(__file__).resolve().parents[1] / 'configs.yaml')
    a = p.parse_args()
    if (a.kl_clean or a.kl_triggered) and not a.backdoored:
        p.error('KL divergence needs --backdoored')
    config = yaml.safe_load(Path(a.config).read_text())
    batch_size, tokens = config['generation']['batch_size'], config['attacks'][a.attack]['eval_tokens']
    a.out.mkdir(parents=True, exist_ok=True)
    scores = {}
    if a.wildguardtest:
        scores['safety'] = safety(a.model, a.wildguardtest, a.out, batch_size)
    if not a.skip_capability:
        scores['capability'] = capability(a.model, batch_size)
    for name, prompts in [('clean', a.kl_clean), ('triggered', a.kl_triggered)]:
        if prompts:
            scores[f'kl_{name}'] = kl(a.backdoored, a.model, prompts, a.out, name, tokens, batch_size)
    if a.attack == 'code':
        scores['coding'] = coding(a.model, batch_size)
    print(json.dumps(scores, indent=2))
    (a.out / 'suite.json').write_text(json.dumps(scores, indent=2) + '\n')


if __name__ == '__main__':
    main()
