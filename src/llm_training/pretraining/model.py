import torch

from einops import einsum, rearrange, reduce


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # softmax on i dim
    # take max on i dim
    # stabilize
    x = x - x.max(dim=dim, keepdim=True).values
    e_x = torch.exp(x)
    return e_x / torch.sum(e_x, dim=dim, keepdim=True)


class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device = None, dtype=None):
        super().__init__()

        # init weights as N(0, 2/d_in + d_out), trunc at 3 std.
        std = (2 / (in_features + out_features)) ** 0.5
        self.W = torch.empty(out_features, in_features, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(self.W, mean=0, std=std, a=-3 * std, b=3 * std)
        # a,b are trunc

        self.W = torch.nn.Parameter(self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, " ... d_in, d_out d_in -> ... d_out")


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device = None, dtype=None):
        super().__init__()

        # init embedding N(0, 1) trunc at +-3.
        self.embedding = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(self.embedding, mean=0, std=1, a=-3, b=3)
        self.embedding = torch.nn.Parameter(self.embedding)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device = None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.gain = torch.ones(d_model, device=device, dtype=dtype)
        self.gain = torch.nn.Parameter(self.gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        ##
        sum_square = reduce(x**2, " ... d_model -> ... 1", "sum")
        rms = torch.sqrt((sum_square / self.d_model) + self.eps)
        return (x * self.gain / rms).to(in_dtype)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device = None, dtype=None):
        super().__init__()

        self.W1, self.W3 = (
            Linear(d_model, d_ff, device=device, dtype=dtype),
            Linear(d_model, d_ff, device=device, dtype=dtype),
        )  # flipped for row major (who knows), i think d_model is always a row.
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(self.silu(self.W1(x)) * self.W3(x))
        # SWILU ABLATION
        # return self.W2(self.silu(self.W1(x)))


class RotaryPositionEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device = None):
        super().__init__()

        idxs = torch.arange(max_seq_len, device=device)
        freqs = theta ** (-torch.arange(0, d_k, 2, device=device) / d_k)
        # freqs go from 0 to d_k in steps of 2.
        angle = torch.outer(idxs, freqs)

        self.register_buffer("cos_cached", torch.cos(angle))
        self.register_buffer("sin_cached", torch.sin(angle))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_cached = self.cos_cached[token_positions]
        sin_cached = self.sin_cached[token_positions]

        x_evens = x[..., 0::2]
        x_odds = x[..., 1::2]

        # we can do without matmul
        x_evens_rot = x_evens * cos_cached - x_odds * sin_cached
        x_odds_rot = x_evens * sin_cached + x_odds * cos_cached

        # what is the shape of x?
        # it is ... d_k

        return rearrange([x_evens_rot, x_odds_rot], "r ... seq d -> ... seq (d r)")


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    QK = einsum(Q, K, " batch_size ... n d_k, batch_size ... m d_k -> batch_size ... n m")

    d_k = K.shape[-1]
    QK_varscale = QK / d_k**0.5

    QK_masked = torch.where(mask, QK_varscale, float("-inf"))

    QK_softmax = softmax(QK_masked, dim=-1)

    return einsum(QK_softmax, V, " batch_size ... n m, batch_size ... m d_v -> batch_size ... n d_v")


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        # equivalent to 3 eparate maps
        self.QKV = Linear(d_model, d_model * 3)

        # O(MHA(Q,K,V))
        self.O = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, pos: torch.Tensor = None, RoPE: RotaryPositionEmbedding = None) -> torch.Tensor:
        # concat version
        qkv = self.QKV(x)

        # unconcat
        q, k, v = qkv.split(self.d_model, dim=2)

        # (batch, seq_len, dim) -> (batch, heads, seq_len, head_dim)
        # split into multi head
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)
        # holy einsum

        seq_length = x.shape[-2]
        if pos is None:
            pos = torch.arange(seq_length, device=x.device)

        if RoPE is None:
            # make identity map
            RoPE = lambda x, pos: x

        q = RoPE(q, pos)
        k = RoPE(k, pos)

        # causal mask
        m = ~torch.triu(torch.ones((seq_length, seq_length), device=x.device, dtype=torch.bool), diagonal=1)

        # pass mask to attention
        y = scaled_dot_product_attention(q, k, v, m)
        # reconcat heads
        y_concat = rearrange(y, "b h s d -> b s (h d)")

        return self.O(y_concat)


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope=None):
        super().__init__()
        # following names in adapters.
        self.rope = rope
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln2 = RMSNorm(d_model)
        # SWiGLU has ffn
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, pos=None):
        # adds keep norm outside of grad path
        ## x += self.attn(self.ln1(x), pos, self.rope)
        # correct:
        x_attn = x + self.attn(self.ln1(x), pos, self.rope)
        return x_attn + self.ffn(self.ln2(x_attn))
        # REMOVE RMSNORM ABLATION
        # x_attn = x + self.attn(x, pos, self.rope)
        # return x_attn + self.ffn(x_attn)
        # REMOVE POSTNORM
        # x_attn = self.ln1(x + self.attn(x, pos, self.rope))
        # return self.ln2(x_attn + self.ffn(x_attn))
        # NoPE ABLATION
        # x_attn = x + self.attn(self.ln1(x), pos, None)
        # return x_attn + self.ffn(self.ln2(x_attn))


class TransformerLM(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, vocab_size, context_length, num_layers, theta=10000.0):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model)

        # vars follow adapters
        # d/h
        rope = RotaryPositionEmbedding(theta, d_model // num_heads, context_length)

        # each layer is zoomed in block pic
        self.layers = torch.nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, rope) for _ in range(num_layers)])

        # final linear + norm
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x, pos=None):
        # index embeddings
        x = self.token_embeddings(x)
        # each layer is block
        for block in self.layers:
            x = block(x, pos)
        # final norm + linear (softmax is elswhere)
        return self.lm_head(self.ln_final(x))
        # REMOVE RMSNORM ABLATION
        # return self.lm_head(x)
