"""Train the deployment handover head from a manifest-backed handover cache."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from vla.adapters.lerobot.checkpoint_contract import sha256_file
from vla.adapters.lerobot.policies.cache_manifest import (
    CACHE_MANIFEST_NAME,
    load_cache_manifest,
)
from vla.adapters.lerobot.policies.dataset_rlt_ac import ChunkTransitionDataset
from vla.cli.build_handover_cache import HANDOVER_CACHE_KIND
from vla.core.handover_checkpoint import (
    HANDOVER_FORMAT_VERSION,
    save_handover_checkpoint,
)
from vla.core.phase_controller import HandoverClassifier


log = logging.getLogger(__name__)


class HandoverCacheDataset(Dataset):
    def __init__(self, cache_dir: str | Path, split: str, *, rl_token_dim: int) -> None:
        self.transitions = ChunkTransitionDataset(cache_dir, split=split)
        self.rl_token_dim = int(rl_token_dim)

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        transition = self.transitions[index]
        if "z_rl" in transition:
            state = transition["z_rl"].detach().float().reshape(-1)
            if state.numel() != self.rl_token_dim:
                raise ValueError(
                    f"cached z_rl has {state.numel()} dims, expected {self.rl_token_dim}"
                )
        else:
            state = transition["state_vec"].detach().float().reshape(-1)
        if state.numel() < self.rl_token_dim:
            raise ValueError(
                f"transition state has {state.numel()} dims, need z_rl={self.rl_token_dim}"
            )
        label = transition["is_critical"].detach().float().reshape(())
        if label.item() not in (0.0, 1.0):
            raise ValueError(f"is_critical must be binary, got {label.item()}")
        return state[: self.rl_token_dim], label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0 or args.hidden_dim <= 0:
        raise ValueError("epochs, batch_size, and hidden_dim must be positive")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("lr must be positive and weight_decay cannot be negative")
    if args.patience <= 0 or args.num_workers < 0:
        raise ValueError("patience must be positive and num_workers cannot be negative")


def _class_counts(dataset: Dataset) -> tuple[int, int]:
    negative = 0
    positive = 0
    for index in range(len(dataset)):
        _, label = dataset[index]
        if label.item() == 1.0:
            positive += 1
        else:
            negative += 1
    if negative == 0 or positive == 0:
        raise ValueError(
            "handover training requires both VLA and critical examples; "
            f"found negative={negative}, positive={positive}"
        )
    return negative, positive


def _binary_metrics(logits: Tensor, labels: Tensor, threshold: float) -> dict[str, float | int]:
    predictions = torch.sigmoid(logits) >= threshold
    truth = labels.bool()
    tp = int((predictions & truth).sum())
    tn = int((~predictions & ~truth).sum())
    fp = int((predictions & ~truth).sum())
    fn = int((~predictions & truth).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return {
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "balanced_accuracy": 0.5 * (recall + tnr),
        "false_positive_rate": fp / max(fp + tn, 1),
        "false_negative_rate": fn / max(fn + tp, 1),
    }


@torch.no_grad()
def _evaluate(
    classifier: HandoverClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
    pos_weight: Tensor,
) -> tuple[float, Tensor, Tensor]:
    classifier.eval()
    losses: list[Tensor] = []
    logits: list[Tensor] = []
    labels: list[Tensor] = []
    for z_rl, target in loader:
        z_rl = z_rl.to(device)
        target = target.to(device)
        output = classifier(z_rl).squeeze(-1)
        losses.append(
            F.binary_cross_entropy_with_logits(
                output, target, pos_weight=pos_weight, reduction="sum"
            ).cpu()
        )
        logits.append(output.cpu())
        labels.append(target.cpu())
    all_logits = torch.cat(logits)
    all_labels = torch.cat(labels)
    return float(torch.stack(losses).sum() / len(all_labels)), all_logits, all_labels


def train_handover(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    manifest_path = cache_dir / CACHE_MANIFEST_NAME
    manifest = load_cache_manifest(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError(f"transition cache is not complete: {manifest_path}")
    policy_contract = manifest["contract"]["policy"]
    cache_kind = manifest["contract"].get("cache", {}).get("kind", "full_transition")
    if cache_kind not in {"full_transition", HANDOVER_CACHE_KIND}:
        raise ValueError(f"unsupported handover training cache kind: {cache_kind!r}")
    rl_token_dim = int(policy_contract["rl_token_dim"])
    rl_token_contract = policy_contract.get("rl_token_contract")
    if not rl_token_contract:
        raise ValueError("transition cache has no RL Token fingerprint")

    train_dataset = HandoverCacheDataset(cache_dir, "train", rl_token_dim=rl_token_dim)
    val_dataset = HandoverCacheDataset(cache_dir, "val", rl_token_dim=rl_token_dim)
    train_counts = _class_counts(train_dataset)
    val_counts = _class_counts(val_dataset)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    device = torch.device(args.device)
    classifier = HandoverClassifier(rl_token_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    pos_weight = torch.tensor(train_counts[0] / train_counts[1], device=device)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        classifier.train()
        train_sum = 0.0
        train_examples = 0
        for z_rl, target in train_loader:
            z_rl = z_rl.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(z_rl).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()
            train_sum += float(loss.detach()) * len(target)
            train_examples += len(target)
        val_loss, _, _ = _evaluate(
            classifier, val_loader, device=device, pos_weight=pos_weight
        )
        train_loss = train_sum / train_examples
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        log.info(
            "handover epoch=%d train_loss=%.6f val_loss=%.6f",
            epoch,
            train_loss,
            val_loss,
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(classifier.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("handover training produced no checkpoint")
    classifier.load_state_dict(best_state, strict=True)
    val_loss, val_logits, val_labels = _evaluate(
        classifier, val_loader, device=device, pos_weight=pos_weight
    )
    metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": val_loss,
        "train_counts": {"negative": train_counts[0], "positive": train_counts[1]},
        "val_counts": {"negative": val_counts[0], "positive": val_counts[1]},
        "at_0_5": _binary_metrics(val_logits, val_labels, 0.5),
        "at_enter_0_8": _binary_metrics(val_logits, val_labels, 0.8),
        "history": history,
    }
    config = {
        "format_version": HANDOVER_FORMAT_VERSION,
        "input_kind": "z_rl",
        "input_dim": rl_token_dim,
        "hidden_dim": args.hidden_dim,
        "label": "is_critical",
        "rl_token_contract": rl_token_contract,
        "cache_manifest_sha256": sha256_file(manifest_path),
        "cache_kind": cache_kind,
        "recommended_enter_threshold": 0.8,
        "recommended_exit_threshold": 0.2,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
        },
    }
    output = save_handover_checkpoint(
        args.output_dir,
        classifier=classifier,
        config=config,
        metrics=metrics,
    )
    return {"output_dir": str(output), "config": config, "metrics": metrics}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = train_handover(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
