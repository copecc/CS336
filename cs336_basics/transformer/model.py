import math
import torch

from torch import nn
from einops import einsum, rearrange, reduce

from cs336_basics.transformer.nn_utils import scaled_dot_product_attention


class Linear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initialize the Linear layer.

        Args:
            in_features (int): The size of the input features.
            out_features (int): The size of the output features.
            device (torch.device | None): The device to create the layer on.
            dtype (torch.dtype | None): The data type of the layer's parameters.
        """
        super().__init__()

        self.d_in = in_features
        self.d_out = out_features

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        std = math.sqrt(2.0 / (self.d_in + self.d_out))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the linear layer.
        """
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initialize the Embedding layer.

        Args:
            num_embeddings (int): The number of embeddings in the vocabulary.
            embedding_dim (int): The size of the embedding dimension.
            device (torch.device | None): The device to create the layer on.
            dtype (torch.dtype | None): The data type of the layer's parameters.
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        std = 1.0
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the embedding layer.

        Select the embedding vector for each token ID by indexing into an embedding matrix.
        """
        return self.weight[token_ids]


class RMSNorm(nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initialize the RMSNorm layer.

        Args:
            d_model (int): The size of the model dimension.
            eps (float): A small value to avoid division by zero.
            device (torch.device | None): The device to create the layer on.
            dtype (torch.dtype | None): The data type of the layer's parameters.
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the RMSNorm layer.
        """
        in_dtype = x.dtype

        x = x.to(torch.float32)
        rms = reduce(x * x, "... d_model -> ... 1", "mean").add(self.eps).sqrt()
        result = x / rms * self.weight
        return result.to(in_dtype)


class SwiGLUFeedForward(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initialize the SwiGLU feed-forward layer.

        Args:
            d_model (int): The size of the model dimension.
            d_ff (int): The size of the feed-forward dimension.
            device (torch.device | None): The device to create the layer on.
            dtype (torch.dtype | None): The data type of the layer's parameters.
        """
        super().__init__()

        if d_ff is None:
            # set dff to approximately 8/3 * d_model and round up to nearest multiple of 64
            target_d_ff = int(8 * d_model / 3)
            d_ff = math.ceil(target_d_ff / 64) * 64

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SwiGLU feed-forward layer.
        """
        w1_x = self.w1(x)
        w3_x = self.w3(x)
        silu = w1_x * torch.sigmoid(w1_x)
        return self.w2(silu * w3_x)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        """
        Initialize the RotaryPositionalEmbedding layer.

        Args:
            theta (float): The rotation angle.
            d_k (int): The dimension of the key vectors.
            max_seq_len (int): The maximum sequence length.
            device (torch.device | None): The device to create the layer on.
        """
        super().__init__()

        # compute the inverse frequencies
        inv_freqs = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        # compute the angles for the position encodings
        angles = torch.outer(positions, inv_freqs)
        # compute and cache the cosine and sine values
        cos_cached = torch.cos(angles)
        sin_cached = torch.sin(angles)

        self.register_buffer("cos_cached", cos_cached.to(device), persistent=False)
        self.register_buffer("sin_cached", sin_cached.to(device), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the RotaryPositionalEmbedding layer.
        """
        # get the cosine and sine values for the current token positions
        # (..., seq_len, d_k//2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        x_even, x_odd = x[..., 0::2], x[..., 1::2]

        # apply rotation
        x_even_rotated = x_even * cos - x_odd * sin
        x_odd_rotated = x_even * sin + x_odd * cos

        x_rotated = torch.empty_like(x)
        x_rotated[..., 0::2] = x_even_rotated
        x_rotated[..., 1::2] = x_odd_rotated

        return x_rotated


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float = None,
        max_seq_len: int = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initialize the MultiHeadSelfAttention layer.

        Args:
            d_model (int): Dimensionality of the input features.
            num_heads (int): Number of attention heads.
            theta (float, optional): Rotary embedding parameter.
            max_seq_len (int, optional): Maximum sequence length.
            device (torch.device, optional): Device to create the layer on.
            dtype (torch.dtype, optional): Data type of the layer parameters.
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        d_k = d_v = d_model // num_heads
        # combining the key, query, and value projections into a single weight matrix so only need a single matrix multiply
        self.qkv_proj = Linear(d_model, 3 * num_heads * d_k, device=device, dtype=dtype)
        self.output_proj = Linear(num_heads * d_v, d_model, device=device, dtype=dtype)

        self.rope = None
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=d_k,
                max_seq_len=max_seq_len,
                device=device,
            )

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass for the multi-head self-attention layer.
        """
        sequence_length = x.shape[-2]
        # create token positions if using rope and not provided
        if self.rope is not None and token_positions is None:
            token_positions = torch.arange(sequence_length, device=x.device)

        # compute query, key, value projections
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(
            qkv,
            "... sequence_length (three num_heads d_k) -> three ... num_heads sequence_length d_k",
            three=3,
            num_heads=self.num_heads,
        )

        if self.rope is not None:
            # apply rotary positional embedding
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        ones = torch.ones(
            sequence_length, sequence_length, dtype=torch.bool, device=x.device
        )
        mask = torch.tril(ones)

        # compute attention scores
        attn_scores = scaled_dot_product_attention(q, k, v, mask)
        attn_scores = rearrange(
            attn_scores,
            "... num_heads sequence_length d_k -> ... sequence_length (num_heads d_k)",
        )

        return self.output_proj(attn_scores)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = None,
        theta: float = None,
        max_seq_len: int = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initialize the Transformer block.

        Args:
            d_model (int): Dimensionality of the input features.
            num_heads (int): Number of attention heads.
            d_ff (int, optional): Dimensionality of the feedforward layer.
            theta (float, optional): Rotary embedding parameter.
            max_seq_len (int, optional): Maximum sequence length.
            device (torch.device, optional): Device to create the layer on.
            dtype (torch.dtype, optional): Data type of the layer parameters.
        """
        super().__init__()

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLUFeedForward(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Transformer block (pre-norm).
        """
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int = None,
        theta: float = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, num_heads, d_ff, theta, context_length, device, dtype
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Transformer language model.
        """
        x = self.token_embeddings(input_ids)

        for block in self.layers:
            x = block(x)

        x = self.ln_final(x)
        return self.lm_head(x)
