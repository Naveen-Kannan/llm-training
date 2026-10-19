import torch, os, json
from llm_training.pretraining.tokenizer import Tokenizer
from llm_training.pretraining.model import softmax, TransformerLM
from llm_training.pretraining.training_utils import load_checkpoint
from llm_training.pretraining.experiment import _pick_device


def next_token(model, input, temp=1.0, top_p=1.0):
    # get rand (0,top_p)
    rand = float(torch.rand(1)) * top_p

    out_logits = model(input)  # only last token
    final_logit = out_logits[:, -1, :]
    # add temp
    probs = softmax(final_logit / temp, dim=-1)[0]
    # sort probs
    probs, ids = torch.sort(probs, descending=True)
    cumsum_probs = torch.cumsum(probs, dim=0)
    # filter for where rand is crossed, 0000011111
    idx = int((cumsum_probs >= rand).nonzero()[0].item())

    return int(ids[idx].item())


def decode(model, prompt, max_tokens, end_token, temp=1.0, top_p=1.0):
    for _ in range(max_tokens - len(prompt)):
        input = prompt
        next_tok = next_token(model, input, temp, top_p)
        # prompt is a tensor? same device?
        prompt = torch.cat([prompt, torch.tensor([[next_tok]], device=prompt.device)], dim=-1)

        if next_tok == end_token:
            break

    return prompt


if __name__ == "__main__":
    # take in arg for checkpoint
    print("Experiment number: ")
    exp_num = input().strip()
    exp_path = f"runs/transformer/exp_{exp_num}"
    ckpt_path = os.path.join(exp_path, "checkpoint.pt")

    # get user prompt
    print("User prompt: ")
    prompt = input()

    # load exp_config
    with open(os.path.join(exp_path, "config.json"), "r") as f:
        exp_config = json.load(f)
        exp_config["device"] = _pick_device()

    data = exp_config["run_config"]["train_data"]
    tokenizer_path = "runs/bpe/"
    if "tinystories" in data:
        tokenizer = Tokenizer.from_files(
            os.path.join(tokenizer_path, "tinystories", "vocab.pkl"),
            os.path.join(tokenizer_path, "tinystories", "merges.pkl"),
        )
    else:
        tokenizer = Tokenizer.from_files(
            os.path.join(tokenizer_path, "openwebtext", "vocab.pkl"),
            os.path.join(tokenizer_path, "openwebtext", "merges.pkl"),
        )
    # tokenize, load tokenizer corresponding to training data in model config/data

    # put in a tensor for model
    prompt_tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=exp_config["device"])

    # load the model
    model = TransformerLM(
        exp_config["model_config"]["d_model"],
        exp_config["model_config"]["num_heads"],
        exp_config["model_config"]["d_ff"],
        exp_config["model_config"]["vocab_size"],
        exp_config["model_config"]["context_length"],
        exp_config["model_config"]["num_layers"],
        exp_config["model_config"]["rope_theta"],
    ).to(exp_config["device"])

    ckpt = torch.load(ckpt_path, map_location=exp_config["device"])
    model.load_state_dict(ckpt["model"])

    # run the model times
    end_tok = tokenizer.inverse_vocab["<|endoftext|>".encode("utf-8")]  # get tok int
    with torch.no_grad():
        decoded_prompt = decode(
            model, prompt_tokens, exp_config["model_config"]["context_length"], end_tok, temp=0.7, top_p=0.9
        )

    # untokenize
    generated_text = tokenizer.decode(decoded_prompt[0].tolist())  # take list of ints

    # print
    print(generated_text)
