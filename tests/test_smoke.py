"""Packaging checks, not a reproduction of GPU training or original grading tests."""

import torch

from llm_training.pretraining.model import TransformerLM
from llm_training.pretraining.tokenizer import Tokenizer
from llm_training.posttraining.grpo_helpers import (
    aggregate_loss_across_microbatch,
    compute_group_normalized_rewards,
    compute_policy_gradient_loss,
)


def test_byte_tokenizer_round_trip():
    vocab = {index: bytes([index]) for index in range(256)}
    vocab[256] = b"hi"
    tokenizer = Tokenizer(vocab, [(b"h", b"i")])
    text = "hi, language models! λ"
    tokens = tokenizer.encode(text)
    assert 256 in tokens
    assert tokenizer.decode(tokens) == text


def test_transformer_cpu_forward_backward():
    torch.manual_seed(0)
    model = TransformerLM(32, 4, 64, 32, 8, 2)
    tokens = torch.randint(0, 32, (2, 8))
    logits = model(tokens)
    assert logits.shape == (2, 8, 32)
    assert torch.isfinite(logits).all()
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 32), tokens.reshape(-1))
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_group_normalization_and_policy_loss_backward():
    advantages, _ = compute_group_normalized_rewards(torch.tensor([0.0, 1.0, 1.0, 0.0]), 2)
    assert torch.allclose(advantages.reshape(2, 2).mean(dim=1), torch.zeros(2))
    log_probs = torch.full((4, 3), -0.5, requires_grad=True)
    token_losses, _ = compute_policy_gradient_loss(advantages[:, None], log_probs)
    loss = aggregate_loss_across_microbatch(token_losses, torch.ones_like(log_probs))
    assert torch.isfinite(loss)
    loss.backward()
    assert log_probs.grad is not None
    assert torch.isfinite(log_probs.grad).all()
