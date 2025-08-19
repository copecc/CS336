import math
import torch

from torch import Tensor, nn
from einops import einsum, rearrange, reduce
from jaxtyping import Float, Bool, Int

from cs336_basics.transformer.nn_utils import silu, scaled_dot_product_attention, top_p_sampling


class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, device: torch.device = None, dtype: torch.dtype = None):
        """
        Linear layer that performs matrix multiplication with learnable weights, initialized using a truncated normal distribution.

        Args:
            in_features (int): The size of the input features.
            out_features (int): The size of the output features.
            device (torch.device): The device to create the layer on.
            dtype (torch.dtype): The data type of the layer's parameters.
        """
        super().__init__()

        self.weight: Float[Tensor, " d_out d_in"] = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        """
        Performs matrix multiplication between the input tensor and the layer's weights.
        """
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

    def extra_repr(self) -> str:
        return f"out_features={self.weight.shape[0]}, in_features={self.weight.shape[1]}"


class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device = None, dtype: torch.dtype = None):
        """
        Embedding layer that maps input token IDs to dense vector representations using a learnable embedding matrix.

        Args:
            num_embeddings (int): The number of embeddings in the vocabulary.
            embedding_dim (int): The size of the embedding dimension.
            device (torch.device): The device to create the layer on.
            dtype (torch.dtype): The data type of the layer's parameters.
        """
        super().__init__()

        self.weight: Float[Tensor, " vocab_size d_model"] = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        std = 1.0
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        """
        Looks up and returns the embedding vectors for the provided token IDs.
        Select the embedding vector for each token ID by indexing into an embedding matrix.
        """
        return self.weight[token_ids]

    def extra_repr(self) -> str:
        return f"num_embeddings(vocabulary size)={self.weight.shape[0]}, embedding_dim={self.weight.shape[1]}"


class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device = None, dtype: torch.dtype = None):
        """
        RMSNorm (Root Mean Square Layer Normalization) normalizes the input tensor and applies a learnable scaling parameter to improve training stability.

        Args:
            d_model (int): The dimensionality of the RMSNorm input.
            eps (float): A value added to the denominator for numerical stability.
            device (torch.device): The device to create the layer on.
            dtype (torch.dtype): The data type of the layer's parameters.
        """
        super().__init__()

        self.eps = eps
        self.weight: Float[Tensor, " d_model"] = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        """
        Normalizes the input tensor using RMS normalization and scales it.
        """
        in_dtype = x.dtype

        x = x.to(torch.float32)
        rms = reduce(x * x, "... d_model -> ... 1", "mean").add(self.eps).rsqrt()
        result = x * rms * self.weight
        return result.to(in_dtype)

    def extra_repr(self) -> str:
        return f"d_model={self.weight.shape[0]}, eps={self.eps}"


class SwiGLUFeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int = None, device: torch.device = None, dtype: torch.dtype = None):
        """
        Feed-forward network using the SwiGLU activation function, consisting of multiple linear transformations and a gating mechanism.

        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
            device (torch.device): The device to create the layer on.
            dtype (torch.dtype): The data type of the layer's parameters.
        """
        super().__init__()

        if d_ff is None:
            # set dff to approximately 8/3 * d_model and round up to nearest multiple of 64
            target_d_ff = int(8 * d_model / 3)
            d_ff = math.ceil(target_d_ff / 64) * 64

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
        """
        Applies two linear transformations and a SwiGLU activation to the input, then projects back to the model dimension.
        """
        return self.w2(silu(self.w1(x)) * self.w3(x))

    def extra_repr(self) -> str:
        return f"d_model={self.w1.in_features}, d_ff={self.w1.out_features}"


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device = None):
        """
        Implements rotary positional embeddings to inject relative positional information into attention mechanisms.

        Args:
            theta (float): The rotation angle.
            d_k (int): The dimension of the key vectors.
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
            device (torch.device): The device to create the layer on.
        """
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for rotary embedding"

        # compute the inverse frequencies
        inv_freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k))
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        # compute the angles for the position encodings
        angles = torch.outer(positions, inv_freqs)
        # compute and cache the cosine and sine values
        cos_cached: Float[Tensor, " ..."] = torch.cos(angles)
        sin_cached: Float[Tensor, " ..."] = torch.sin(angles)

        self.register_buffer("cos_cached", cos_cached.to(device), persistent=False)
        self.register_buffer("sin_cached", sin_cached.to(device), persistent=False)

    def forward(
        self, x: Float[Tensor, " ... sequence_length d_k"], token_positions: Int[Tensor, " ... sequence_length"]
    ) -> Float[Tensor, " ... sequence_length d_k"]:
        """
        Applies rotary positional encoding to the input tensor based on token positions.
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

    def extra_repr(self) -> str:
        return f"context_length={self.cos_cached.shape[0]}, d_k/2={self.cos_cached.shape[1]}"


class CausalMultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        """
        Multi-head self-attention layer with causal masking and optional rotary positional embeddings.

        This module projects the input into queries, keys, and values, applies multi-head self-attention with a causal mask (preventing each position from attending to future positions), and supports rotary positional embeddings for improved relative position encoding. The output is projected back to the model dimension.

        Args:
            d_model (int): Hidden size of the model (the main feature dimension used throughout the Transformer).
            num_heads (int): Number of heads to use in multi-headed attention.
            rope (RotaryPositionalEmbedding, optional): Rotary positional embedding module.
            device (torch.device): The device to create the layer on.
            dtype (torch.dtype): The data type of the layer's parameters.
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        d_k = d_v = d_model // num_heads
        # combining the key, query, and value projections into a single weight matrix so only need a single matrix multiply
        self.qkv_proj = Linear(d_model, 3 * num_heads * d_k, device=device, dtype=dtype)
        self.output_proj = Linear(num_heads * d_v, d_model, device=device, dtype=dtype)

        self.rope = rope

    def forward(
        self, x: Float[Tensor, "... sequence_length d_in"], token_positions: Int[Tensor, " ... sequence_length"] = None
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        """
        Applies multi-head self-attention with causal masking and optional rotary positional embeddings.

        This method projects the input into queries, keys, and values, applies rotary positional encoding if enabled.
        Performs scaled dot-product attention with a causal mask to prevent attending to future positions.
        Finally, projects the concatenated attention outputs back to the model dimension.
        """
        sequence_length = x.shape[-2]

        # compute query, key, value projections
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "... seq (three heads d_k) -> three ... heads seq d_k", three=3, heads=self.num_heads)

        # create token positions if not provided
        if token_positions is None:
            token_positions = torch.arange(sequence_length, device=x.device)
        # apply rotary positional embedding if provided
        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        ones = torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=x.device)
        causal_mask = torch.tril(ones)

        # compute attention scores
        attn_scores = scaled_dot_product_attention(q, k, v, causal_mask)
        attn_scores = rearrange(attn_scores, "... heads seq d_k -> ... seq (heads d_k)")

        return self.output_proj(attn_scores)

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, d_model={self.qkv_proj.weight.shape[-1]}"


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = None,
        theta: float = None,
        max_seq_len: int = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        """
        The basic Transformer block, composed of pre-layer normalization, multi-head self-attention, and a SwiGLU feed-forward network, with residual connections.

        Args:
            d_model (int): The dimensionality of the Transformer block input.
            num_heads (int): Number of attention heads.
            d_ff (int, optional): Dimensionality of the feedforward layer.
            theta (float, optional): Rotary embedding parameter.
            max_seq_len (int, optional): Maximum sequence length.
            device (torch.device): The device to create the layer on.
            dtype (torch.dtype): The data type of the layer's parameters.
        """
        super().__init__()

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)

        rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device=device)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, rope, device=device, dtype=dtype)

        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLUFeedForward(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self, x: Float[Tensor, " batch sequence_length d_model"]
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        """
        Applies layer normalization, self-attention, and feed-forward network with residual connections (pre-norm).
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
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        """
        Transformer-based language model that stacks multiple Transformer blocks, takes token IDs as input, and outputs logits for each token position.

        Args:
            vocab_size (int): Size of the vocabulary.
            context_length (int): Maximum context length for input sequences.
            num_layers (int): Number of Transformer layers.
            d_model (int): Hidden size of the model (the main feature dimension used throughout the Transformer).
            num_heads (int): Number of attention heads.
            d_ff (int, optional): Dimensionality of the feedforward layer.
            theta (float, optional): Rotary embedding parameter.
            device (torch.device, optional): The device to create the layer on.
            dtype (torch.dtype, optional): The data type of the layer's parameters.
        """
        super().__init__()

        self.context_length = context_length

        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, theta, context_length, device, dtype)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embeds input token IDs, passes them through stacked Transformer blocks and final normalization, then projects to vocabulary logits.
        """
        x = self.token_embeddings(input_ids)

        for block in self.layers:
            x = block(x)

        x = self.ln_final(x)
        return self.lm_head(x)

    def get_num_parameters(self, non_embedding: bool = True) -> int:
        """
        Returns the total number of trainable parameters in the model.

        If non_embedding is True (default), the returned parameter count excludes the last layer (output layer).
        This is because, in many research papers and model reports, the output layer parameters are often reported separately from the rest of the model
        (sometimes embeddings are *shared* or tied with the embedding layer).
        """
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Exclude output layer parameters
        if non_embedding:
            num_params -= self.lm_head.weight.numel()
        return num_params

    @torch.no_grad()
    def generate(
        self,
        x: Int[Tensor, " ... sequence_length"],
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = None,
        top_k: int = None,
        eos_token_id: int = None,
    ) -> Int[Tensor, " ... max_new_tokens"]:
        """
        Generates new tokens from the model given an input sequence.

        Args:
            x (torch.Tensor): Input tensor of shape (1, sequence_length) or (sequence_length, ) containing token IDs.
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Controls randomness in sampling. Higher values produce more random outputs.
            top_p (float, optional): If specified, limits the sampling to the top p cumulative probability.
            top_k (int, optional): If specified, limits the sampling to the top k logits.
            eos_token_id (int, optional): If specified, generation stops when this token is produced.

        Returns:
            torch.Tensor: Tensor of shape (1, max_new_tokens) containing generated token IDs.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        original_length = x.shape[1]
        for _ in range(max_new_tokens):
            x = x[:, -self.context_length :] if x.shape[1] > self.context_length else x
            logits = self.forward(x)
            next_token_logits = logits[:, -1, :] / temperature

            if top_p is not None:  # Apply top-p (nucleus) sampling
                next_token = top_p_sampling(next_token_logits, p=top_p, num_samples=1)
            else:
                if top_k is not None:  # Apply top-k sampling
                    top_k_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    threshold = top_k_values[:, -1]
                    mask = next_token_logits < threshold
                    next_token_logits.masked_fill_(mask, float("-inf"))

                next_token_probabilities = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probabilities, num_samples=1)

            x = torch.cat((x, next_token), dim=1)
            # Check for end-of-sequence token
            if eos_token_id is not None and (next_token == eos_token_id).any():
                break

        new_tokens = x[:, original_length:]
        return new_tokens
