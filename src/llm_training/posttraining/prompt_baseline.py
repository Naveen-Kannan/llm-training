import json
from pathlib import Path
from llm_training.posttraining.vllm_utils import VLLMServer
from llm_training.posttraining.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from collections import defaultdict


def parse_ans(ans):
    return ans.split("####")[-1].strip()


def eval(prompt_type, prompts, sampling_params):
    completions = llm.generate_completions(prompts, sampling_params, batch_size=32)

    if prompt_type == "question_only":
        reward = question_only_reward_fn
    else:
        reward = r1_zero_reward_fn

    categories = defaultdict(list)
    for i in range(len(completions)):
        ex, completion = gsm8k[i], completions[i].text
        ans = parse_ans(ex["answer"])
        r = reward(completion, ans)
        if int(r["format_reward"]):
            if int(r["answer_reward"]):
                categories["formatted_and_correct"].append((ex, completion))
            else:
                categories["formatted_and_incorrect"].append((ex, completion))
        else:
            categories["unformatted"].append((ex, completion))

    for cat in ["formatted_and_correct", "formatted_and_incorrect", "unformatted"]:
        print(cat, len(categories[cat]))

    # print 10 examples each category
    print(prompt_type, "--------------------------------")
    for category, examples in categories.items():
        print(category, ":")
        for ex, completion in examples[:10]:
            print(ex["question"])
            print(completion)
            print("-")


if __name__ == "__main__":
    # data in path data/gsm8k/test.jsonl
    data_path = Path("data/gsm8k/test.jsonl")
    with open(data_path, "r") as f:
        gsm8k = [json.loads(line) for line in f]
    # load prompts
    with open(Path("src/llm_training/posttraining/prompts/question_only.prompt"), "r") as f:
        question_only_prompt = f.read()
    with open(Path("src/llm_training/posttraining/prompts/r1_zero.prompt"), "r") as f:
        r1_zero_prompt = f.read()

    llm = VLLMServer(model_id="allenai/OLMo-2-0425-1B")
    llm.start()

    sampling_params = {"temperature": 1.0, "max_tokens": 512, "n": 1, "seed": 0}  # n is top p (?)
    prompts_qo = [question_only_prompt.format(question=ex["question"]) for ex in gsm8k]
    eval("question_only", prompts_qo, sampling_params.copy())

    sampling_params["stop"] = ["</answer>"]
    sampling_params["include_stop_str_in_output"] = True
    # r1 zero eval
    prompts_r1_0 = [r1_zero_prompt.format(question=ex["question"]) for ex in gsm8k]
    eval("r1_zero", prompts_r1_0, sampling_params.copy())

    # r1 one eval
    with open(Path("src/llm_training/posttraining/prompts/r1_zero_three_shot_gsm8k.prompt")) as f:
        r1_three_shot_prompt = f.read()
    prompts_r1_3 = [r1_three_shot_prompt.format(question=ex["question"]) for ex in gsm8k]
    eval("r1_zero_three_shot", prompts_r1_3, sampling_params.copy())
    llm.stop()
