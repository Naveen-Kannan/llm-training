from llm_training.posttraining.grpo_helpers import grpo_train_step, tokenize_prompt_and_output, get_response_log_probs
from pathlib import Path
from llm_training.posttraining.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from llm_training.posttraining.vllm_utils import VLLMServer
from llm_training.posttraining.prompt_baseline import parse_ans
from llm_training.posttraining.checkpoint import get_model_and_tokenizer
import torch
import json, random, wandb, os
import numpy as np


# typical
tbs = 256
gas = 32

# EXP CONFIGS
# seed  = 50
# exp_name = 'drgrpo_resp_len_penalty'

# normal
# loss_agg = 'sequence'
# adv_norm_type = 'std'
# baseline_type = 'mean'

# # grpo constant
# loss_agg = 'constant'
# adv_norm_type = 'std'
# baseline_type = 'mean'

# # dr grpo
# loss_agg = 'constant'
# adv_norm_type = 'none'
# baseline_type = 'mean'

# # dr grpo w/ resp len penalty
# loss_agg = 'constant'
# adv_norm_type = 'none'
# baseline_type = 'mean'
# resp_len_penalty = True

# # rft
# loss_agg = 'constant'
# adv_norm_type = 'none'
# baseline_type = 'none'

# # maxrl
# loss_agg = 'constant'
# adv_norm_type = 'mean'
# baseline_type = 'mean'

#### OFF POLICY
tbs = 8
gas = 1
exp_name = "offpolicy_naive"
loss_agg = "sequence"
adv_norm_type = "std"
baseline_type = "mean"
resp_len_penalty = False
importance_reweighting_method = "none"
cliprange = 0.2
# cliprange = 3e-4

# # noclip
# exp_name = 'offpolicy_noclip'
# importance_reweighting_method = 'noclip'

# # clip
# exp_name = 'offpolicy_clip'
# importance_reweighting_method = 'grpo'

# gspo
exp_name = "offpolicy_gspo"
importance_reweighting_method = "gspo"
cliprange = 3e-4


n_train_examples = 6400
n_val_examples = 1024
num_rollout_steps = 200
# num_rollout_steps = 50
learning_rate = 1e-5
rollout_batch_size = 256
train_batch_size = tbs
group_size = 8
gradient_accumulation_steps = gas
sampling_temperature = 1.0
sampling_max_tokens = 512
max_grad_norm = 1.0
prompt_name = "r1_zero"
val_iter = 10
train_log_iter = 40
n_prompts_per_rollout_batch = rollout_batch_size // group_size
model = "allenai/OLMo-2-0425-1B"
train_file_path = Path("data/gsm8k/train.jsonl")
val_file_path = Path("data/gsm8k/test.jsonl")
log_file_path = Path(f"llm_training.posttraining/experiments/{exp_name}.log")
os.makedirs(log_file_path.parent, exist_ok=True)

normalization_constant = rollout_batch_size * sampling_max_tokens

# tuh


# if __name__ == "__main__":
def main(seed):
    sampling_params = {
        "temperature": sampling_temperature,
        "max_tokens": sampling_max_tokens,
        "n": group_size,
        "seed": seed,
    }
    if prompt_name != "question_only":
        sampling_params["stop"] = ["</answer>"]
        sampling_params["include_stop_str_in_output"] = True

    # global seed
    wandb.init(
        project=f"llm_training.posttraining",
        name=f"seed_{seed}",
        group=f"{exp_name}",
        config={
            "seed": seed,
            "prompt": prompt_name,
            "model": model,
        },
    )

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # load data
    with open(train_file_path, "r") as f:
        train_ex = [json.loads(line) for line in f]
        # 6400 examples
        train_ex = random.sample(train_ex, n_train_examples)
    with open(val_file_path, "r") as f:
        val_ex = [json.loads(line) for line in f]
        # 1024 examples
        # val_ex = random.sample(val_ex, n_val_examples)

    with open(Path(f"src/llm_training/posttraining/prompts/{prompt_name}.prompt"), "r") as f:
        prompt_template = f.read()

    # load; gpu 0 for optimizer, 1 for server
    policy, tokenizer = get_model_and_tokenizer(model, device="cuda:0")
    # policy = policy.float()
    llm = VLLMServer(model_id=model, gpu=1)
    llm.start()
    llm.init_weight_sync("cuda:0")
    optimizer = torch.optim.AdamW(  # 0
        policy.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0
    )

    # tuhh
    for step in range(num_rollout_steps):
        seed = seed + step
        llm.sync_policy_weights(policy)

        batch_ex = random.sample(train_ex, n_prompts_per_rollout_batch)
        batch_prompts = [prompt_template.format(question=ex["question"]) for ex in batch_ex]
        batch_ans = [parse_ans(ex["answer"]) for ex in batch_ex]

        batch_completions = llm.generate_completions(batch_prompts, sampling_params, batch_size=32)

        full_batch_prompts = list()
        full_batch_rollouts = list()
        full_batch_ans = list()
        for p in range(n_prompts_per_rollout_batch):
            for g in range(group_size):
                comp = batch_completions[p * group_size + g]
                full_batch_prompts.append(batch_prompts[p])
                full_batch_rollouts.append(comp.text)
                full_batch_ans.append(batch_ans[p])

        # full_batch_prompts = [None] * rollout_batch_size
        # full_batch_rollouts = [None] * rollout_batch_size
        # full_batch_ans = [None] * rollout_batch_size
        # # swap index order, rollout with new seeds
        # for g in range(group_size):
        #     rollout_sampling_params = sampling_params.copy()
        #     rollout_sampling_params['n'] = 1 # one
        #     rollout_sampling_params['seed'] = seed + g + step * rollout_batch_size
        #     group_completions = llm.generate_completions(batch_prompts, rollout_sampling_params, batch_size=32)
        #     for p in range(n_prompts_per_rollout_batch):
        #         idx = p*group_size + g
        #         full_batch_prompts[idx] = batch_prompts[p]
        #         full_batch_rollouts[idx] = group_completions[p].text
        #         full_batch_ans[idx] = batch_ans[p]

        reward_fn = question_only_reward_fn if prompt_name == "question_only" else r1_zero_reward_fn

        iterations = rollout_batch_size // train_batch_size  # should only be even

        if importance_reweighting_method != "none":
            # # retokenize, cant return from step
            # tokens = tokenize_prompt_and_output(full_batch_prompts, full_batch_rollouts, tokenizer)
            # inp_ids, labels = tokens['input_ids'], tokens['labels']
            old_log_probs_parts = []
            with torch.no_grad():
                for k in range(iterations):
                    batch_start, batch_end = k * train_batch_size, k * train_batch_size + train_batch_size
                    tokens = tokenize_prompt_and_output(
                        full_batch_prompts[batch_start:batch_end], full_batch_rollouts[batch_start:batch_end], tokenizer
                    )
                    inp_ids_chunk = tokens["input_ids"].to("cuda:0")
                    labels_chunk = tokens["labels"].to("cuda:0")
                    chunk = get_response_log_probs(policy, inp_ids_chunk, labels_chunk)
                    # free gpu space
                    old_log_probs_parts.append(chunk["log_probs"].cpu())
            old_log_probs = old_log_probs_parts
        else:
            old_log_probs = None

        tot_loss = 0.0
        metadata = {}
        for k in range(iterations):
            batch_start, batch_end = k * train_batch_size, k * train_batch_size + train_batch_size

            batch_prompts = full_batch_prompts[batch_start:batch_end]
            batch_rollouts = full_batch_rollouts[batch_start:batch_end]
            batch_ans = full_batch_ans[batch_start:batch_end]

            if old_log_probs is not None:
                old_log_probs_batch = old_log_probs[k]
            else:
                old_log_probs_batch = None

            loss, metadata = grpo_train_step(
                policy,
                tokenizer,
                optimizer,
                gradient_accumulation_steps,
                max_grad_norm,
                reward_fn,
                batch_prompts,
                batch_rollouts,
                batch_ans,
                group_size,
                baseline=baseline_type,
                advantage_normalizer=adv_norm_type,
                importance_reweighting_method=importance_reweighting_method,
                old_log_probs=old_log_probs_batch,
                cliprange=cliprange,
                loss_normalization=loss_agg,
                normalization_constant=normalization_constant,
                resp_len_penalty=resp_len_penalty,
            )
            # ideally accum metadata but im lazy

            tot_loss += loss.item()
        loss = torch.tensor(tot_loss / iterations)

        log_itr = {
            "loss": loss.item(),
            "grad_norm": float(metadata["gradient_norm"]),
            "tok_entropy": float(metadata["token_entropy"]),
            "reward_mean": float(metadata["reward_mean"]),
            "format_reward_mean": float(metadata["format_reward_mean"]),
            # 'max_group_mean': metadata['max_group_mean'],
            # 'min_group_mean': metadata['min_group_mean'],
            # 'max_group_std': metadata['max_group_std'],
            # 'min_group_std': metadata['min_group_std'],
        }

        if resp_len_penalty:
            log_itr["mean_length_penalty"] = float(metadata["r_meta"].get("mean_length_penalty", 0))

        itr = step + 1
        if itr % val_iter == 0:
            # update log
            # ug
            val_batch_ex = random.sample(val_ex, n_val_examples)
            val_batch_prompts = [prompt_template.format(question=ex["question"]) for ex in val_batch_ex]
            val_sampleing_params = sampling_params.copy()
            val_sampleing_params["n"] = 1  # not gorup
            llm.sync_policy_weights(policy)
            val_batch_completions = llm.generate_completions(val_batch_prompts, val_sampleing_params, batch_size=32)

            # full_val_prompts = list()
            # full_val_rollouts = list()
            # full_val_ans = list()
            val_rewards = list()
            val_format_rewards = list()
            val_resp_lens = list()
            for j in range(len(val_batch_completions)):
                # full_val_prompts.append(val_batch_prompts[j])
                # full_val_rollouts.append(val_batch_completions[j].text)
                # full_val_ans.append(parse_ans(val_batch_prompts[j]))
                val_ans = parse_ans(val_batch_ex[j]["answer"])
                reward = reward_fn(val_batch_completions[j].text, val_ans)
                val_rewards.append(reward["reward"])
                val_format_rewards.append(reward["format_reward"])
                val_resp_lens.append(len(val_batch_completions[j].token_ids))

            log_itr["val_reward"] = np.average(val_rewards)
            log_itr["val_format_reward"] = np.average(val_format_rewards)
            log_itr["val_resp_len"] = np.average(val_resp_lens)

            print("val_reward", log_itr["val_reward"])

        if itr % train_log_iter == 0:
            # log 5 examples from full_batch_prompts, full_batch_rollouts, full_batch_ans
            for j in range(5):
                log_itr[f"prompt_{j}"] = full_batch_prompts[j]
                log_itr[f"rollout_{j}"] = full_batch_rollouts[j]
                log_itr[f"ans_{j}"] = full_batch_ans[j]
                # i think all?
                print("prompt", full_batch_prompts[j])
                print("rollout", full_batch_rollouts[j])
                print("ans", full_batch_ans[j])

        with open(log_file_path, "a") as f:
            f.write(json.dumps(log_itr) + "\n")

        wandb.log(log_itr, step=itr)

    llm.stop()
    wandb.finish()


if __name__ == "__main__":
    for seed in [0, 1, 2, 3]:
        main(seed)
