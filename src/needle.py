"""NEEDLE: training-free backdoor removal that preserves refusal.

NEEDLE estimates a backdoor direction b_l and a rank-4 refusal subspace R_l at every layer, then
edits the model layer by layer: it removes b_l from each layer's output weights
while keeping their refusal projections fixed, and corrects the MLP output weights so that the
activations keep their original refusal projections.

Sections
  1. Activation capture
  2. Backdoor direction (Eq. 3)
  3. Refusal subspace (Eqs. 4-5)
  4. Weight orthogonalisation (Eq. 6)
  5. Correction (Eqs. 7-9)
  6. Sequential edit (applies 4 and 5 to each edited layer in order)
"""
import torch

# 1. Capture activations

def decoder_layers(model):
    """Return the language model's decoder blocks in order (vision layers are skipped)."""
    found = {int(name.split('.layers.')[1]): module for name, module in model.named_modules()
             if '.layers.' in name and name.split('.layers.')[1].isdigit() and 'vision' not in name}
    assert sorted(found) == list(range(len(found)))
    return [found[i] for i in range(len(found))]


def copied_prompt_tokens(tokenizer, prompt, response_ids):
    """Count leading response tokens that repeat the prompt (and a following 'Answer:').

    These tokens are excluded when estimating the refusal subspace.
    """
    text = tokenizer.decode(response_ids, skip_special_tokens=True)
    prompt = prompt.strip()
    start = len(text) - len(text.lstrip())
    if not prompt or not text[start:].startswith(prompt):
        return 0
    stop = start + len(prompt)
    while stop < len(text) and text[stop].isspace():
        stop += 1
    if text[stop:stop + 7].casefold() == 'answer:':
        stop += 7
        while stop < len(text) and text[stop].isspace():
            stop += 1
    count = 0
    for k in range(1, len(response_ids) + 1):  # longest token prefix that stays within the copy
        if len(tokenizer.decode(response_ids[:k], skip_special_tokens=True)) > stop:
            break
        count = k
    return count


@torch.inference_mode()
def response_activations(model, tokenizer, histories, skip_copied_prompt=False):
    """Mean block output over each history's response tokens, at every layer.

    Special tokens are excluded; with skip_copied_prompt, a leading copy of the prompt is too.
    Returns a float32 tensor of shape [histories, layers, hidden].
    """
    blocks, special = decoder_layers(model), set(tokenizer.all_special_ids)
    result = []
    for row in histories:
        ids, response = row['input_token_ids'], row['generated_token_ids']
        start = copied_prompt_tokens(tokenizer, row['prompt'], response) if skip_copied_prompt else 0
        positions = [len(ids) + i for i, t in enumerate(response) if i >= start and t not in special]
        assert positions, f"no usable response tokens in {row.get('id')}"
        means = [None] * len(blocks)

        def hook(layer):
            def record(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                means[layer] = h[0, positions].float().mean(0).cpu()
            return record

        handles = [block.register_forward_hook(hook(i)) for i, block in enumerate(blocks)]
        try:
            model(input_ids=torch.tensor([ids + response], device=model.device), use_cache=False, logits_to_keep=1)
        finally:
            for handle in handles:
                handle.remove()
        result.append(torch.stack(means))
    return torch.stack(result)


def observation_positions(row, special):
    """Final prompt token and the response tokens at 1/4, 2/4 and 3/4 of the response."""
    prompt, response = row['input_token_ids'], row['generated_token_ids']
    valid = [len(prompt) + i for i, t in enumerate(response) if t not in special]
    return [len(prompt) - 1] + [valid[(len(valid) - 1) * k // 4] for k in (1, 2, 3)]


@torch.inference_mode()
def observe(model, tokenizer, histories, layers, R):
    """Record, at the observation positions, what the correction needs at each given layer.

    Per layer: MLP down-projection inputs 'x' and outputs 'y' (float32), and the refusal
    projections R_l^T h of the block output 'z' (float64), with four rows per history.
    """
    blocks, special = decoder_layers(model), set(tokenizer.all_special_ids)
    store = {l: {'x': [], 'y': [], 'z': []} for l in layers}
    positions = []

    def mlp_hook(layer):
        def record(module, args, output):
            store[layer]['x'].append(args[0][0, positions].float().cpu())
            store[layer]['y'].append(output[0, positions].float().cpu())
        return record

    def block_hook(layer):
        def record(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            store[layer]['z'].append((h[0, positions].double() @ R[layer].to(h.device)).cpu())
        return record

    handles = []
    for l in layers:
        handles.append(blocks[l].mlp.down_proj.register_forward_hook(mlp_hook(l)))
        handles.append(blocks[l].register_forward_hook(block_hook(l)))
    try:
        for row in histories:
            positions[:] = observation_positions(row, special)
            ids = torch.tensor([row['input_token_ids'] + row['generated_token_ids']], device=model.device)
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    return {l: {k: torch.cat(v) for k, v in values.items()} for l, values in store.items()}

# 2. Backdoor direction (Eq. 3)

def backdoor_direction(clean, triggered):
    """Unit direction from clean to triggered mean activations, orthogonal to the clean mean.

    clean, triggered: [pairs, layers, hidden] response activations. Returns b [layers, hidden].
    For targeted refusal, 'triggered' holds triggered prompts followed by the clean responses.
    """
    mu_clean, mu_triggered = clean.double().mean(0), triggered.double().mean(0)
    u = mu_clean / mu_clean.norm(dim=-1, keepdim=True)
    delta = mu_triggered - mu_clean
    b = delta - (delta * u).sum(-1, keepdim=True) * u
    return b / b.norm(dim=-1, keepdim=True)

# 3. Refusal subspace (Eqs. 4-5)

def refusal_subspace(refused, complied, rank=4):
    """Orthonormal refusal basis R [layers, hidden, rank] from paired refused/compliant activations.

    The first column is the refusal direction; the rest are the leading directions of variation
    among the paired differences.
    """
    refused, complied = refused.double(), complied.double()
    return torch.stack([_refusal_basis(refused[:, l], complied[:, l], rank) for l in range(refused.shape[1])])


def _refusal_basis(h, c, rank):
    """Refusal basis at one layer; h and c are [pairs, hidden]."""
    delta = h.mean(0) - c.mean(0)
    mid = (h.mean(0) + c.mean(0)) / 2
    v = mid / mid.norm(dim=-1, keepdim=True)
    r = delta - (delta * v).sum(-1, keepdim=True) * v  # refusal direction, orthogonal to the midpoint
    r = r / r.norm(dim=-1, keepdim=True)
    residual = h - c - delta  # centred paired differences
    residual = residual - (residual @ v)[:, None] * v
    residual = residual - (residual @ r)[:, None] * r
    _, singular, vt = torch.linalg.svd(residual, full_matrices=False)
    assert singular[rank - 2] > 1e-8, 'not enough variation for the requested rank'
    R, _ = torch.linalg.qr(torch.cat([r[:, None], vt[:rank - 1].T], 1))
    if R[:, 0] @ r < 0:  # keep the refusal direction's sign
        R[:, 0] *= -1
    return R

# 4. Weight orthogonalisation (Eq. 6)

def orthogonalise(weight, b, R):
    """W1 = W - u (b^T W) / (b^T u), with u = (I - R R^T) b.

    Removes the backdoor direction from the matrix's outputs (b^T W1 = 0) while leaving their
    refusal projections unchanged (R^T W1 = R^T W). Computed in float64 on the CPU.
    """
    W, b, R = weight.detach().cpu().double(), b.cpu().double(), R.cpu().double()
    u = b - R @ (R.T @ b)
    denominator = b @ u  # equals ||u||^2
    assert float(denominator) > 1e-8, 'backdoor direction lies inside the refusal subspace'
    return (W - (u / denominator)[:, None] * (b @ W)[None, :]).to(weight.dtype)

# 5. Correction (Eqs. 7-9)

def correction(x, y, error, R, b, norm=None, ridge=1e-3):
    """Low-rank update to an MLP output matrix that restores the activations' refusal projections.

    x: [n, intermediate] MLP down-projection inputs; y: [n, hidden] its outputs; error: [n, k]
    original minus current refusal projections. The update writes only within U = (I - b b^T) R,
    so it never reintroduces the backdoor direction. The ridge penalty is `ridge` times the mean
    diagonal of the Gram matrix. `norm` = (gain, eps) linearises a normalisation applied to the
    MLP output (Gemma's post-feedforward RMSNorm); it is None when there is none.
    """
    x, y, error, R, b = (t.double() for t in (x, y, error, R, b))
    U = R - b[:, None] * (b @ R)[None, :]
    if norm is None:
        JU = U.unsqueeze(0).expand(len(y), -1, -1)
    else:  # Jacobian of RMSNorm at each observed output, applied to U
        gain, eps = norm
        s2 = y.square().mean(-1) + eps
        tangent = U[None] - y[:, :, None] * (y @ U)[:, None, :] / (y.shape[1] * s2[:, None, None])
        JU = gain.double()[None, :, None] / s2.sqrt()[:, None, None] * tangent
    G = torch.einsum('dk,ndj->nkj', R, JU)  # per observation: update coefficients -> refusal projections
    c = torch.einsum('nkj,nj->nk', torch.linalg.pinv(G, rtol=1e-10), error)  # coefficients that remove the error
    gram = x @ x.T
    penalty = ridge * gram.diag().mean()
    C = x.T @ torch.linalg.solve(gram + penalty * torch.eye(len(x), device=x.device, dtype=x.dtype), c)  # ridge, dual form
    return U @ C.T


def mlp_output_norm(block):
    """(gain, eps) of a normalisation applied to the MLP output, or None if the block has none."""
    norm = getattr(block, 'post_feedforward_layernorm', None)
    if norm is None:
        return None
    return 1 + norm.weight.detach().double(), float(norm.eps)  # Gemma's RMSNorm scales by (1 + weight)

# 6. Sequential edit

def needle_edit(model, tokenizer, b, R, histories, target_histories=None, layers=None, ridge=1e-3):
    """Apply NEEDLE to the model in place.

    For each edited layer in increasing order: orthogonalise its attention and MLP output weights
    (Eq. 6), then correct its MLP output weights (Eqs. 7-9) using activations of the model edited
    so far. Targets are the refusal projections of the unedited model on `target_histories`
    (default: `histories`). Layers default to floor(L/3) .. L-1.
    """
    blocks = decoder_layers(model)
    lo, hi = layers if layers is not None else (len(blocks) // 3, len(blocks))
    b, R = b.double(), R.double()
    targets = observe(model, tokenizer, target_histories or histories, range(lo, hi), R)
    for layer in range(lo, hi):
        block = blocks[layer]
        with torch.no_grad():
            for linear in (block.self_attn.o_proj, block.mlp.down_proj):
                linear.weight.copy_(orthogonalise(linear.weight, b[layer], R[layer]).to(linear.weight.device))
        current = observe(model, tokenizer, histories, [layer], R)[layer]
        down = block.mlp.down_proj
        device = down.weight.device
        delta = correction(current['x'].to(device), current['y'].to(device),
                           (targets[layer]['z'] - current['z']).to(device),
                           R[layer].to(device), b[layer].to(device), mlp_output_norm(block), ridge)
        with torch.no_grad():
            down.weight.copy_((down.weight.double() + delta).to(down.weight.dtype))
        print(f'edited layer {layer + 1}/{len(blocks)}', flush=True)
    return model
