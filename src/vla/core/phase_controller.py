from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn


class Phase(Enum):
    VLA_PHASE = "vla_phase"
    CRITICAL_PHASE = "critical_phase"


class HandoverClassifier(nn.Module):
    """Binary classifier on z_rl to predict critical state (for deployment)."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_rl: torch.Tensor) -> torch.Tensor:
        """Returns logit for critical phase prediction, (B, 1)."""
        return self.net(z_rl)

    def predict(self, z_rl: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Returns boolean tensor (B,) indicating critical phase."""
        return torch.sigmoid(self.forward(z_rl)).squeeze(-1) > threshold


class PhaseController:
    """State machine for VLA -> RL handoff.

    Manual mode (training): human triggers transition.
    Learned mode (deployment): binary classifier on z_rl predicts critical state.
    """

    def __init__(
        self,
        mode: str = "manual",
        classifier: HandoverClassifier | None = None,
        enter_threshold: float = 0.8,
        exit_threshold: float = 0.2,
        debounce_chunks: int = 2,
        manual_cooldown_chunks: int = 3,
    ):
        if mode not in ("manual", "learned"):
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.classifier = classifier
        if not 0 <= exit_threshold < enter_threshold <= 1:
            raise ValueError("phase thresholds must satisfy 0 <= exit < enter <= 1")
        if debounce_chunks <= 0 or manual_cooldown_chunks < 0:
            raise ValueError("debounce must be positive and manual cooldown cannot be negative")
        self.enter_threshold = float(enter_threshold)
        self.exit_threshold = float(exit_threshold)
        self.debounce_chunks = int(debounce_chunks)
        self.manual_cooldown_chunks = int(manual_cooldown_chunks)
        self._phase = Phase.VLA_PHASE
        self._enter_votes = 0
        self._exit_votes = 0
        self._manual_cooldown_remaining = 0
        self.last_probability: float | None = None

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def is_critical(self) -> bool:
        return self._phase == Phase.CRITICAL_PHASE

    def reset(self) -> None:
        """Reset to VLA phase at episode start."""
        self._phase = Phase.VLA_PHASE
        self._enter_votes = 0
        self._exit_votes = 0
        self._manual_cooldown_remaining = 0
        self.last_probability = None

    @property
    def manual_cooldown_remaining(self) -> int:
        return self._manual_cooldown_remaining

    def _after_manual_change(self) -> None:
        self._enter_votes = 0
        self._exit_votes = 0
        if self.mode == "learned":
            self._manual_cooldown_remaining = self.manual_cooldown_chunks

    def trigger_critical(self) -> None:
        """Manual trigger: switch to critical phase."""
        self._phase = Phase.CRITICAL_PHASE
        self._after_manual_change()

    def trigger_vla(self) -> None:
        """Manual trigger: switch back to VLA phase."""
        self._phase = Phase.VLA_PHASE
        self._after_manual_change()

    def update(self, z_rl: torch.Tensor | None = None) -> Phase:
        """Update phase based on mode.

        In manual mode, phase only changes via trigger_critical/trigger_vla.
        In learned mode, classifier decides based on z_rl.

        Args:
            z_rl: (B, D) RL token embedding (required for learned mode)

        Returns:
            current phase
        """
        if self.mode == "learned":
            if self.classifier is None:
                raise RuntimeError("Learned mode requires a classifier")
            if z_rl is None:
                raise ValueError("Learned mode requires z_rl")
            if self._manual_cooldown_remaining > 0:
                self._manual_cooldown_remaining -= 1
                return self._phase
            probability = torch.sigmoid(self.classifier(z_rl)).squeeze(-1)
            if probability.numel() != 1:
                raise ValueError("learned phase control requires inference batch size 1")
            self.last_probability = float(probability.item())
            if self._phase == Phase.VLA_PHASE:
                self._exit_votes = 0
                if self.last_probability >= self.enter_threshold:
                    self._enter_votes += 1
                else:
                    self._enter_votes = 0
                if self._enter_votes >= self.debounce_chunks:
                    self._phase = Phase.CRITICAL_PHASE
                    self._enter_votes = 0
            else:
                self._enter_votes = 0
                if self.last_probability <= self.exit_threshold:
                    self._exit_votes += 1
                else:
                    self._exit_votes = 0
                if self._exit_votes >= self.debounce_chunks:
                    self._phase = Phase.VLA_PHASE
                    self._exit_votes = 0

        return self._phase

    def train_classifier(
        self,
        z_rl: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Single training step for the handover classifier.

        Args:
            z_rl: (B, D) RL token embeddings
            labels: (B,) binary labels (1 = critical)
            optimizer: optimizer for classifier params

        Returns:
            loss value
        """
        if self.classifier is None:
            raise RuntimeError("No classifier to train")
        logits = self.classifier(z_rl).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
