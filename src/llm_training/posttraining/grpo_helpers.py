import torch, math
from collections import defaultdict

# get model and tokenizer
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# def get_model_and_tokenizer(model_id_or_dir: str, device: str):
# model = AutoModelForCausalLM.from_pretrained(
# model_id_or_dir,
# device_map=device,
# torch_dtype=torch.bfloat16,
# attn_implementation="eager" if device=='cpu' else "flash_attention_2",
# )
# tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
# return model, tokenizer


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    tokens = defaultdict(list)
    prompt_and_output_lens = 0

    for prompt, output in zip(prompt_strs, output_strs):
        p_toks, o_toks = (
            tokenizer.encode(prompt, add_special_tokens=False),
            tokenizer.encode(output, add_special_tokens=False),
        )
        all_toks = p_toks + o_toks
        prompt_and_output_lens = max(prompt_and_output_lens, len(all_toks))
        tokens["all_toks"].append(all_toks)
        tokens["prompt_lens"].append(len(p_toks))

    # padding
    for all_toks, prompt_len in zip(tokens["all_toks"], tokens["prompt_lens"]):
        filling = prompt_and_output_lens - len(all_toks)
        all_toks_padded = all_toks + [tokenizer.pad_token_id] * filling
        tokens["input_ids_fr"].append(all_toks_padded[:-1])
        tokens["labels_fr"].append(all_toks_padded[1:])

        mask = list()
        for j in range(len(all_toks_padded) - 1):  # label aligned
            if j + 1 >= prompt_len and j + 1 < len(all_toks):
                mask.append(1)
            else:
                mask.append(0)

        tokens["response_mask_fr"].append(mask)

    tokens["input_ids"] = torch.tensor(tokens["input_ids_fr"])
    tokens["labels"] = torch.tensor(tokens["labels_fr"])
    tokens["response_mask"] = torch.tensor(tokens["response_mask_fr"])

    del tokens["input_ids_fr"]
    del tokens["labels_fr"]
    del tokens["response_mask_fr"]

    return {"input_ids": tokens["input_ids"], "labels": tokens["labels"], "response_mask": tokens["response_mask"]}


def get_response_log_probs(model, input_ids, labels, return_token_entropy=False):
    input_ids = input_ids.to(model.device)
    labels = labels.to(model.device)
    # with torch.no_grad():
    logits = model(input_ids=input_ids, labels=labels).logits
    # # should be fine to do torch softmax now
    # probs = torch.softmax(logits, dim=-1)
    # log_probs = torch.log(probs)
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    # get log _probs to be for the nxt tok
    log_probs_tok = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))  # dope
    log_probs_tok = log_probs_tok.squeeze(-1)

    ret = dict()
    if return_token_entropy:
        # exp them now
        ret["token_entropy"] = -(log_probs.exp() * log_probs).sum(dim=-1)

    ret["log_probs"] = log_probs_tok
    return ret


def compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths):
    # rollout_batch_size = len(repeated_ground_truths)
    # # group_size = len(rollout_responses) // rollout_batch_size
    # n_prompts_per_rollout_batch = rollout_batch_size // group_size

    rewards = []
    # for i in range(group_size):
    #     # gt = repeated_ground_truths[i]
    #     for j in range(n_prompts_per_rollout_batch):
    #         gt = repeated_ground_truths[i* n_prompts_per_rollout_batch + j]
    #         response = rollout_responses[i* n_prompts_per_rollout_batch + j]
    #         r = reward_fn(response, gt)
    #         rewards.append(r)
    for i in range(len(rollout_responses)):
        gt = repeated_ground_truths[i]
        response = rollout_responses[i]
        r = reward_fn(response, gt)
        rewards.append(r)

    format_rewards = [r["format_reward"] for r in rewards]
    rewards = [r["reward"] for r in rewards]

    format_rewards = torch.tensor(format_rewards)
    rewards = torch.tensor(rewards)

    metadata = {
        "mean_total_reward": torch.mean(rewards),
        "mean_format_reward": torch.mean(format_rewards),
    }

    return (rewards, metadata)


def compute_group_normalized_rewards(
    raw_rewards, group_size, baseline="mean", advantage_eps=1e-6, advantage_normalizer="std"
):
    # rearrange by groups einops
    # nvm cant use einops?
    raw_rewards_matrix = raw_rewards.view(-1, group_size)
    group_means = torch.mean(raw_rewards_matrix, dim=1)
    group_stds = torch.std(raw_rewards_matrix, dim=1)
    group_means_flat = group_means.view(-1, 1)
    group_stds_flat = group_stds.view(-1, 1)

    if baseline == "mean":
        raw_rewards_matrix = raw_rewards_matrix - group_means_flat
    elif baseline == "none":
        raw_rewards_matrix = raw_rewards_matrix

    if advantage_normalizer == "std":
        raw_rewards_matrix = raw_rewards_matrix / (group_stds_flat + advantage_eps)
    elif advantage_normalizer == "none":
        raw_rewards_matrix = raw_rewards_matrix
    elif advantage_normalizer == "mean":
        raw_rewards_matrix = raw_rewards_matrix / (group_means_flat + advantage_eps)

    metadata = {
        "max_group_mean": torch.max(group_means),
        "min_group_mean": torch.min(group_means),
        "max_group_std": torch.max(group_stds),
        "min_group_std": torch.min(group_stds),
    }

    return (raw_rewards_matrix.view(-1), metadata)


def compute_policy_gradient_loss(
    raw_rewards_or_advantages,
    policy_log_probs,
    importance_reweighting_method="none",
    old_log_probs=None,
    cliprange=None,
    response_mask=None,
):
    # if importance_reweighting_method != 'none':
    #     raise NotImplementedError

    metadata = dict()

    if importance_reweighting_method == "none":
        # not not summed over seq yet
        per_token_policy_gradient_loss = -raw_rewards_or_advantages * policy_log_probs
    elif importance_reweighting_method == "noclip":
        imp_ratio = torch.exp(policy_log_probs - old_log_probs)
        per_token_policy_gradient_loss = -raw_rewards_or_advantages * imp_ratio
    elif importance_reweighting_method == "grpo":
        imp_ratio = torch.exp(policy_log_probs - old_log_probs)
        # # init mask as zeros
        # mask = torch.zeros_like(policy_log_probs, dtype=torch.bool)
        # # set 1 if imp ratio is < 1 + cliprange and adv >= 0
        # mask = ((imp_ratio < 1 + cliprange) & (raw_rewards_or_advantages >= 0))
        # # same for other
        # mask |= ((imp_ratio > 1 - cliprange) & (raw_rewards_or_advantages < 0))
        # per_token_policy_gradient_loss = -raw_rewards_or_advantages *
        # policy_log_probs * imp_ratio * mask
        clipped_imp_ratio = torch.clamp(imp_ratio, 1 - cliprange, 1 + cliprange)
        per_token_policy_gradient_loss = -torch.min(
            raw_rewards_or_advantages * imp_ratio, raw_rewards_or_advantages * clipped_imp_ratio
        )
    elif importance_reweighting_method == "gspo":
        # in lgo form
        all_log_ratios = policy_log_probs - old_log_probs
        # exp_coeff = torch.max(1, response_mask.sum(dim=1, keepdim=True))
        exp_coeff = torch.clamp(response_mask.sum(dim=1, keepdim=True), min=1)
        imp_ratio = torch.exp((all_log_ratios * response_mask).sum(dim=1, keepdim=True) / exp_coeff)

        clipped_imp_ratio = torch.clamp(imp_ratio, 1 - cliprange, 1 + cliprange)
        per_token_policy_gradient_loss = -torch.min(
            raw_rewards_or_advantages * imp_ratio, raw_rewards_or_advantages * clipped_imp_ratio
        )

        per_token_policy_gradient_loss = per_token_policy_gradient_loss.repeat(1, policy_log_probs.shape[1])
    else:
        raise NotImplementedError

    return (per_token_policy_gradient_loss, metadata)


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss, mask, loss_normalization="sequence", normalization_constant=None
):
    if loss_normalization == "sequence":
        per_token_policy_gradient_loss = per_token_policy_gradient_loss * mask
        # num not masked
        num_not_masked = torch.sum(mask, dim=1)
        # B, S; B includes groups, doesnt matter?
        tot_seq_loss = torch.sum(per_token_policy_gradient_loss, dim=1)
        avg_seq_loss = tot_seq_loss / num_not_masked
        batch_size = per_token_policy_gradient_loss.shape[0]
        avg_loss = torch.sum(avg_seq_loss) / batch_size
        # avg_loss = torch.mean(avg_seq_loss)
    elif loss_normalization == "constant":
        # tot_seq_loss = torch.sum(per_token_policy_gradient_loss, dim=1)
        # # avg_seq_loss = tot_seq_loss / normalization_constant
        # # batch_size = per_token_policy_gradient_loss.shape[0]
        # # avg_loss = torch.sum(avg_seq_loss) /batch_size
        # avg_loss = tot_seq_loss / normalization_constant
        per_token_policy_gradient_loss = per_token_policy_gradient_loss * mask
        avg_loss = torch.sum(per_token_policy_gradient_loss) / normalization_constant
    else:
        raise NotImplementedError

    return avg_loss


def grpo_train_step(
    model,
    tokenizer,
    optimizer,
    gradient_accumulation_steps,
    max_grad_norm,
    reward_fn,
    repeated_prompts,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    baseline="mean",
    advantage_eps=1e-6,
    advantage_normalizer="std",
    importance_reweighting_method="none",
    old_log_probs=None,
    cliprange=None,
    loss_normalization="sequence",
    normalization_constant=None,
    resp_len_penalty=False,
):
    prompt_count = len(repeated_prompts)
    pre_prune_microbatch_size = prompt_count // gradient_accumulation_steps
    optimizer.zero_grad()
    # tot_loss, train_reward, format_reward, tok_entropy = 0.0, 0.0, 0.0, 0.0
    # not imp?
    raw_rewards, r_meta = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    train_reward = raw_rewards.sum().item()

    # modification attempt 1
    if resp_len_penalty:
        min_resp_len = 50
        max_penalty = 0.05
        resp_lens = torch.tensor(
            [len(tokenizer.encode(r, add_special_tokens=False)) for r in rollout_responses],
            dtype=torch.float32,
        )
        penalty = max_penalty * torch.clamp((min_resp_len - resp_lens) / min_resp_len, min=0)
        # lets only do it if no reward
        raw_rewards = raw_rewards - penalty * (raw_rewards == 0).float()
        r_meta["mean_length_penalty"] = penalty.mean().item()

    advantages, a_meta = compute_group_normalized_rewards(
        raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer
    )
    # train_reward = raw_rewards.sum().item()
    # format_upscale =
    format_reward = r_meta["mean_format_reward"].item() * len(rollout_responses)

    # filter adv
    new_repeated_prompts = []
    new_rollout_responses = []
    new_repeated_ground_truths = []
    new_advantages = []
    for a_idx in range(len(advantages)):
        if abs(advantages[a_idx]) > 0.0:
            new_repeated_prompts.append(repeated_prompts[a_idx])
            new_rollout_responses.append(rollout_responses[a_idx])
            new_repeated_ground_truths.append(repeated_ground_truths[a_idx])
            new_advantages.append(advantages[a_idx])

    repeated_prompts = new_repeated_prompts
    rollout_responses = new_rollout_responses
    repeated_ground_truths = new_repeated_ground_truths
    # tensor
    advantages = torch.tensor(new_advantages)

    new_prompt_count = len(repeated_prompts)

    if new_prompt_count == 0:
        return torch.tensor(0.0), {
            "r_meta": r_meta,
            "a_meta": a_meta,
            "l_metadata": {},
            "gradient_norm": 0.0,
            "token_entropy": 0.0,
            "reward_mean": train_reward / prompt_count,
            "format_reward_mean": format_reward / prompt_count,
        }

    # round up
    gradient_accumulation_steps = math.ceil(gradient_accumulation_steps * new_prompt_count / prompt_count)

    microbatch_size = math.ceil(new_prompt_count / gradient_accumulation_steps)

    tot_loss, tok_entropy = 0.0, 0.0
    n_tok = 0

    tokens = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    all_input_ids, all_labels, all_response_mask = tokens["input_ids"], tokens["labels"], tokens["response_mask"]
    # next get
    device = next(model.parameters()).device

    for i in range(gradient_accumulation_steps):
        s, e = i * microbatch_size, (i + 1) * microbatch_size
        # get microbatch
        input_ids_microbatch = all_input_ids[s:e]
        labels_microbatch = all_labels[s:e]
        response_mask_microbatch = all_response_mask[s:e]
        advantages_microbatch = advantages[s:e]
        # mvoe everything on device, next get
        input_ids = input_ids_microbatch.to(device)
        labels = labels_microbatch.to(device)
        response_mask = response_mask_microbatch.to(device)
        advantages_microbatch = advantages_microbatch.to(device)

        n_tok += response_mask.sum().item()
        probs = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
        log_probs = probs["log_probs"]
        tok_entropy += (probs["token_entropy"] * response_mask).sum().item()
        off_policy_probs = None
        if importance_reweighting_method != "none":
            off_policy_probs = old_log_probs[s:e].to(device)
        per_token_loss, l_metadata = compute_policy_gradient_loss(
            advantages_microbatch.unsqueeze(-1),
            log_probs,
            importance_reweighting_method,
            off_policy_probs,
            cliprange,
            response_mask,
        )
        agg_loss = aggregate_loss_across_microbatch(
            per_token_loss, response_mask, loss_normalization, normalization_constant
        )
        if loss_normalization == "constant":
            upscaled_loss = agg_loss
        else:
            upscaled_loss = agg_loss * pre_prune_microbatch_size / prompt_count
        upscaled_loss.backward()  # auto accum
        tot_loss += float(upscaled_loss.item())
        # [0]

    gradient_norm_pre_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    metadata = {
        "r_meta": r_meta,
        "a_meta": a_meta,
        "l_metadata": l_metadata,
        "gradient_norm": gradient_norm_pre_clip,
        "token_entropy": tok_entropy / n_tok,
        "reward_mean": train_reward / prompt_count,
        "format_reward_mean": format_reward / prompt_count,
    }

    # if resp_len_penalty:
    #     metadata['mean_length_penalty'] = r_meta['mean_length_penalty']

    # val
    return torch.tensor(tot_loss), metadata
