"""A sequence classifier that handles variable-length transaction histories."""

from __future__ import annotations

from numbers import Integral, Real

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class ScalperLSTM(nn.Module):
    """Return one uncalibrated scalper logit per right-padded history.

    Only the first ``lengths[i]`` rows of each history are observed by the LSTM.
    Padding therefore cannot change the prediction for a shorter history.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 48,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        for name, value in (
            ("input_size", input_size),
            ("hidden_size", hidden_size),
            ("num_layers", num_layers),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(dropout, bool) or not isinstance(dropout, Real) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be a finite number in [0, 1)")

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(self.hidden_size, 1)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Classify ``features[batch, time, input_size]`` using valid lengths."""
        if not isinstance(features, torch.Tensor) or features.ndim != 3:
            raise ValueError("features must be a tensor shaped [batch, time, input_size]")
        if not features.is_floating_point():
            raise ValueError("features must contain floating-point values")
        if features.shape[0] < 1 or features.shape[1] < 1:
            raise ValueError("features must contain at least one nonempty history")
        if features.shape[2] != self.input_size:
            raise ValueError(f"features must have {self.input_size} input columns")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("features must contain only finite values")
        if not isinstance(lengths, torch.Tensor) or lengths.ndim != 1:
            raise ValueError("lengths must be a one-dimensional integer tensor")
        if lengths.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("lengths must contain integers")
        if lengths.shape[0] != features.shape[0]:
            raise ValueError("lengths must contain one value per history")
        if bool(((lengths < 1) | (lengths > features.shape[1])).any()):
            raise ValueError("each length must be between 1 and the padded history length")

        packed = pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.classifier(self.dropout(hidden[-1])).squeeze(-1)
