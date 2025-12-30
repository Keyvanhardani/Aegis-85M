# EFC Training Script
# Architecture verification with TinyShakespeare
# Reference: Phase 2 Training Pipeline (docs/architecture.md)

import os
import sys
import math
import time
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import EFCModel, create_efc_model
from src.data import TinyShakespeare


@dataclass
class TrainConfig:
    """Training configuration."""
    # Data
    data_dir: str = "./data"
    seq_len: int = 256
    val_split: float = 0.1

    # Model (small config for verification)
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_state: int = 8
    d_conv: int = 4
    expand: int = 2
    window_size: int = 64
    attention_layers: Optional[list] = None  # Every 4th layer by default

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    epochs: int = 10
    warmup_steps: int = 100

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 1  # Save every N epochs
    log_every: int = 50  # Log every N steps

    # Device
    device: str = "cpu"

    def __post_init__(self):
        """Set attention layers if not specified."""
        if self.attention_layers is None:
            # Every 4th layer gets attention
            self.attention_layers = [
                i for i in range(self.n_layers) if (i + 1) % 4 == 0
            ]


class TrainingLogger:
    """
    Training metrics logger.

    Tracks loss, perplexity, gradient norms, and training speed.
    Outputs to console and saves to JSON.
    """

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history = []
        self.current_epoch_metrics = []

    def log_step(
        self,
        step: int,
        epoch: int,
        loss: float,
        grad_norm: float,
        lr: float,
        tokens_per_sec: float,
    ) -> None:
        """Log single training step."""
        perplexity = math.exp(min(loss, 100))  # Cap to prevent overflow

        metric = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "perplexity": perplexity,
            "grad_norm": grad_norm,
            "lr": lr,
            "tokens_per_sec": tokens_per_sec,
            "timestamp": time.time(),
        }

        self.current_epoch_metrics.append(metric)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        elapsed: float,
    ) -> None:
        """Log epoch summary."""
        train_ppl = math.exp(min(train_loss, 100))
        val_ppl = math.exp(min(val_loss, 100))

        summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_perplexity": train_ppl,
            "val_loss": val_loss,
            "val_perplexity": val_ppl,
            "elapsed_seconds": elapsed,
            "step_metrics": self.current_epoch_metrics,
        }

        self.metrics_history.append(summary)
        self.current_epoch_metrics = []

        print(f"\n[EPOCH {epoch}] Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.2f}")
        print(f"  Time: {elapsed:.1f}s")

    def save(self, filename: str = "training_log.json") -> None:
        """Save metrics to JSON."""
        filepath = self.log_dir / filename
        with open(filepath, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"[LOG] Saved training log to {filepath}")


class Trainer:
    """
    EFC Model Trainer.

    Handles complete training pipeline:
    - Data loading
    - Model initialization
    - Training loop with gradient clipping
    - Validation
    - Checkpointing
    - Metrics logging
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_training()
        self._setup_checkpointing()

        self.logger = TrainingLogger(log_dir=config.checkpoint_dir)
        self.global_step = 0

    def _setup_data(self) -> None:
        """Initialize dataset and data loaders."""
        print("[INIT] Loading TinyShakespeare dataset...")
        self.dataset = TinyShakespeare(
            data_dir=self.config.data_dir,
            seq_len=self.config.seq_len,
            val_split=self.config.val_split,
        )

        self.train_loader = self.dataset.get_train_loader(
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        self.val_loader = self.dataset.get_val_loader(
            batch_size=self.config.batch_size,
        )

    def _setup_model(self) -> None:
        """Initialize model."""
        print("[INIT] Creating EFC model...")

        model_config = {
            "vocab_size": self.dataset.vocab_size,
            "d_model": self.config.d_model,
            "n_layers": self.config.n_layers,
            "n_heads": self.config.n_heads,
            "d_state": self.config.d_state,
            "d_conv": self.config.d_conv,
            "expand": self.config.expand,
            "max_seq_len": self.config.seq_len,
            "attention_layers": self.config.attention_layers,
            "window_size": self.config.window_size,
        }

        self.model = create_efc_model(model_config)
        self.model.to(self.device)

        params = self.model.count_parameters()
        print(f"[INIT] Model parameters: {params['total_millions']:.2f}M")
        print(f"[INIT] Attention layers: {self.config.attention_layers}")

    def _setup_training(self) -> None:
        """Initialize optimizer and scheduler."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )

        # Cosine annealing with warmup
        total_steps = len(self.train_loader) * self.config.epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.learning_rate * 0.1,
        )

        print(f"[INIT] Optimizer: AdamW (lr={self.config.learning_rate})")
        print(f"[INIT] Total steps: {total_steps}")

    def _setup_checkpointing(self) -> None:
        """Setup checkpoint directory."""
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INIT] Checkpoint dir: {self.checkpoint_dir}")

    def _compute_grad_norm(self) -> float:
        """Compute gradient norm across all parameters."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return math.sqrt(total_norm)

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            leave=False,
        )

        for batch_idx, (input_ids, labels) in enumerate(pbar):
            step_start = time.time()

            # Move to device
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, labels=labels)
            loss = outputs["loss"]

            # Backward pass
            loss.backward()

            # Gradient clipping
            grad_norm = self._compute_grad_norm()
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()

            # Metrics
            loss_val = loss.item()
            total_loss += loss_val
            num_batches += 1
            self.global_step += 1

            # Tokens per second
            step_time = time.time() - step_start
            tokens_per_sec = (
                input_ids.numel() / step_time if step_time > 0 else 0
            )

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "ppl": f"{math.exp(min(loss_val, 100)):.2f}",
                "grad": f"{grad_norm:.3f}",
            })

            # Logging
            if self.global_step % self.config.log_every == 0:
                self.logger.log_step(
                    step=self.global_step,
                    epoch=epoch,
                    loss=loss_val,
                    grad_norm=grad_norm,
                    lr=self.scheduler.get_last_lr()[0],
                    tokens_per_sec=tokens_per_sec,
                )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    @torch.no_grad()
    def validate(self) -> float:
        """
        Run validation.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for input_ids, labels in self.val_loader:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(input_ids, labels=labels)
            total_loss += outputs["loss"].item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "config": asdict(self.config),
        }

        # Save latest
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)

        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, epoch_path)

        print(f"[CHECKPOINT] Saved: {epoch_path}")

    def load_checkpoint(self, path: str) -> int:
        """
        Load checkpoint and return starting epoch.

        Args:
            path: Path to checkpoint file

        Returns:
            Epoch to resume from
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]

        print(f"[CHECKPOINT] Loaded: {path}")
        print(f"[CHECKPOINT] Resuming from epoch {checkpoint['epoch']}")

        return checkpoint["epoch"]

    @torch.no_grad()
    def generate_sample(self, prompt: str = "ROMEO:", max_tokens: int = 100) -> str:
        """Generate sample text for qualitative evaluation."""
        self.model.eval()

        # Encode prompt
        input_ids = self.dataset.encode(prompt).unsqueeze(0).to(self.device)

        # Generate
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_k=40,
        )

        # Decode
        generated = self.dataset.decode(output_ids[0])
        return generated

    def train(self) -> None:
        """Run full training loop."""
        print("\n" + "=" * 60)
        print("[TRAIN] Starting EFC Training Pipeline")
        print("=" * 60)

        best_val_loss = float("inf")

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()

            # Training
            train_loss = self.train_epoch(epoch)

            # Validation
            val_loss = self.validate()

            # Epoch metrics
            elapsed = time.time() - epoch_start
            self.logger.log_epoch(epoch, train_loss, val_loss, elapsed)

            # Checkpointing
            if epoch % self.config.save_every == 0:
                self.save_checkpoint(epoch, val_loss)

            # Track best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = self.checkpoint_dir / "checkpoint_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_loss": val_loss,
                }, best_path)
                print(f"[BEST] New best validation loss: {val_loss:.4f}")

            # Generate sample
            if epoch % 2 == 0:
                print("\n[SAMPLE] Generated text:")
                print("-" * 40)
                sample = self.generate_sample()
                print(sample[:300])
                print("-" * 40)

        # Save final metrics
        self.logger.save()

        print("\n" + "=" * 60)
        print("[TRAIN] Training Complete!")
        print(f"[TRAIN] Best validation loss: {best_val_loss:.4f}")
        print(f"[TRAIN] Best perplexity: {math.exp(best_val_loss):.2f}")
        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EFC Model Training - Architecture Verification"
    )

    # Data args
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--seq-len", type=int, default=256)

    # Model args
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)

    # Training args
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None)

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    # Create config
    config = TrainConfig(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        grad_clip=args.grad_clip,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    # Initialize trainer
    trainer = Trainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        # Adjust epoch range if resuming
        config.epochs = config.epochs - start_epoch + 1

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
