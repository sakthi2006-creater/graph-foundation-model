"""
Adapter module for few-shot fine-tuning with frozen encoders.

Standalone implementations of adapter layers that can be used with
any pre-trained encoder for efficient few-shot transfer learning.
"""

from typing import Optional

import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    """
    Simple bottleneck adapter with residual connection.

    Maps: hidden_dim -> adapter_dim -> hidden_dim with residual.

    Args:
        hidden_dim (int): Embedding dimension.
        adapter_dim (int): Bottleneck dimension. Default: 64.
        dropout_p (float): Dropout probability. Default: 0.1.
        activation (str): Activation ('relu', 'gelu'). Default: 'relu'.

    Attributes:
        hidden_dim (int): Embedding dimension.
        adapter_dim (int): Bottleneck dimension.
        down_project (nn.Linear): Project to bottleneck.
        up_project (nn.Linear): Project back to embedding space.
    """

    def __init__(
        self,
        hidden_dim: int,
        adapter_dim: int = 64,
        dropout_p: float = 0.1,
        activation: str = "relu",
    ) -> None:
        """Initialize bottleneck adapter."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.adapter_dim = adapter_dim

        # Activation
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Bottleneck layers
        self.down_project = nn.Linear(hidden_dim, adapter_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.up_project = nn.Linear(adapter_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adapter with residual.

        Args:
            x (torch.Tensor): Input of shape (*, hidden_dim).

        Returns:
            torch.Tensor: Output of shape (*, hidden_dim).
                Computed as: x + adapter(x)
        """
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_project(x)
        return x + residual


class CompactAdapter(nn.Module):
    """
    Compact adapter designed for minimal parameter overhead.

    Uses factorized low-rank decomposition for extreme compression:
    hidden_dim -> rank -> hidden_dim

    Args:
        hidden_dim (int): Embedding dimension.
        rank (int): Low-rank dimension. Default: 8.
        dropout_p (float): Dropout probability. Default: 0.1.

    Attributes:
        hidden_dim (int): Embedding dimension.
        rank (int): Low-rank dimension.
        down (nn.Linear): Hidden to rank.
        up (nn.Linear): Rank to hidden.
    """

    def __init__(
        self,
        hidden_dim: int,
        rank: int = 8,
        dropout_p: float = 0.1,
    ) -> None:
        """Initialize compact adapter."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank

        # Low-rank decomposition
        self.down = nn.Linear(hidden_dim, rank)
        self.dropout = nn.Dropout(p=dropout_p)
        self.up = nn.Linear(rank, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply compact adapter with residual.

        Args:
            x (torch.Tensor): Input of shape (*, hidden_dim).

        Returns:
            torch.Tensor: Output of shape (*, hidden_dim).
        """
        residual = x
        x = self.down(x)
        x = self.dropout(x)
        x = self.up(x)
        return x + residual


class LoRAAdapter(nn.Module):
    """
    Low-Rank Adaptation (LoRA) for efficient fine-tuning.

    Decomposes weight update as A @ B where A and B are low-rank matrices.
    Updates: output = x + (x @ A) @ B * alpha / rank

    Args:
        hidden_dim (int): Embedding dimension.
        rank (int): LoRA rank. Default: 8.
        alpha (float): LoRA alpha (scaling). Default: 16.
        dropout_p (float): Dropout on input before LoRA. Default: 0.1.

    Attributes:
        hidden_dim (int): Embedding dimension.
        rank (int): LoRA rank.
        alpha (float): LoRA scaling factor.
        lora_a (nn.Linear, optional): LoRA matrix A (disabled by default).
        lora_b (nn.Linear, optional): LoRA matrix B (disabled by default).
    """

    def __init__(
        self,
        hidden_dim: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout_p: float = 0.1,
    ) -> None:
        """Initialize LoRA adapter."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.dropout = nn.Dropout(p=dropout_p)

        # LoRA matrices (initialized randomly)
        self.lora_a = nn.Linear(hidden_dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, hidden_dim, bias=False)

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_a.weight, a=torch.nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA adaptation.

        Args:
            x (torch.Tensor): Input of shape (*, hidden_dim).

        Returns:
            torch.Tensor: Output of shape (*, hidden_dim).
                Computed as: x + scaling * (x @ A) @ B
        """
        # LoRA forward: x @ A @ B * (alpha / rank)
        lora_out = self.lora_b(self.lora_a(self.dropout(x)))
        return x + lora_out * self.scaling


class PrefixAdapter(nn.Module):
    """
    Prefix tuning adapter for prompt-based fine-tuning.

    Adds learnable prefix vectors that are prepended to the sequence,
    affecting attention computations without modifying the model parameters.

    Args:
        hidden_dim (int): Embedding dimension.
        prompt_len (int): Length of prefix sequence. Default: 10.
        num_heads (int): Number of attention heads. Default: 8.
        dropout_p (float): Dropout probability. Default: 0.1.

    Attributes:
        prompt_embeddings (nn.ParameterList): Learnable prefix vectors.
    """

    def __init__(
        self,
        hidden_dim: int,
        prompt_len: int = 10,
        num_heads: int = 8,
        dropout_p: float = 0.1,
    ) -> None:
        """Initialize prefix adapter."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.prompt_len = prompt_len
        self.num_heads = num_heads

        # Learnable prefix tokens for each layer
        # shape: (prompt_len, hidden_dim)
        self.prefix = nn.Parameter(torch.randn(prompt_len, hidden_dim))
        nn.init.normal_(self.prefix, std=0.02)

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply prefix adapter by prepending learnable vectors.

        Args:
            x (torch.Tensor): Input of shape (*, seq_len, hidden_dim).

        Returns:
            torch.Tensor: Output with prefix prepended,
                shape (*, seq_len + prompt_len, hidden_dim).
        """
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]

        # Repeat prefix for batch
        prefix = self.prefix.unsqueeze(0)  # (1, prompt_len, hidden_dim)
        prefix = prefix.expand(*batch_shape, -1, -1)  # (batch, prompt_len, hidden_dim)
        prefix = self.dropout(prefix)

        # Concatenate prefix with input
        output = torch.cat([prefix, x], dim=-2)  # (batch, prompt_len + seq_len, hidden_dim)

        return output


class AdapterModule(nn.Module):
    """
    Unified adapter interface supporting multiple strategies.

    Factory-like class that provides a standard interface for different
    adapter types while maintaining compatibility with existing code.

    Args:
        hidden_dim (int): Embedding dimension.
        adapter_type (str): Type of adapter ('bottleneck', 'compact', 'lora', 'prefix').
            Default: 'bottleneck'.
        **kwargs: Type-specific arguments (adapter_dim, rank, alpha, prompt_len, etc.).

    Attributes:
        adapter (nn.Module): The underlying adapter implementation.
        adapter_type (str): Type of adapter used.
    """

    def __init__(
        self,
        hidden_dim: int,
        adapter_type: str = "bottleneck",
        **kwargs,
    ) -> None:
        """Initialize adapter module."""
        super().__init__()
        self.adapter_type = adapter_type.lower()

        if self.adapter_type == "bottleneck":
            self.adapter = BottleneckAdapter(hidden_dim=hidden_dim, **kwargs)
        elif self.adapter_type == "compact":
            self.adapter = CompactAdapter(hidden_dim=hidden_dim, **kwargs)
        elif self.adapter_type == "lora":
            self.adapter = LoRAAdapter(hidden_dim=hidden_dim, **kwargs)
        elif self.adapter_type == "prefix":
            self.adapter = PrefixAdapter(hidden_dim=hidden_dim, **kwargs)
        else:
            raise ValueError(
                f"Unknown adapter type: {self.adapter_type}. "
                f"Supported: ['bottleneck', 'compact', 'lora', 'prefix']"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adapter transformation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Adapted output.
        """
        return self.adapter(x)

    @staticmethod
    def count_parameters(hidden_dim: int, adapter_type: str = "bottleneck", **kwargs) -> int:
        """
        Count number of parameters for different adapter types.

        Args:
            hidden_dim (int): Embedding dimension.
            adapter_type (str): Adapter type.
            **kwargs: Type-specific arguments.

        Returns:
            int: Number of parameters.
        """
        if adapter_type.lower() == "bottleneck":
            adapter_dim = kwargs.get("adapter_dim", 64)
            return 2 * hidden_dim * adapter_dim + adapter_dim + hidden_dim

        elif adapter_type.lower() == "compact":
            rank = kwargs.get("rank", 8)
            return 2 * hidden_dim * rank + rank + hidden_dim

        elif adapter_type.lower() == "lora":
            rank = kwargs.get("rank", 8)
            return 2 * hidden_dim * rank

        elif adapter_type.lower() == "prefix":
            prompt_len = kwargs.get("prompt_len", 10)
            return prompt_len * hidden_dim

        else:
            return 0


# Alias for backward compatibility
AdapterModel = BottleneckAdapter
