#!/usr/bin/env python3
"""
AEGIS-85M Interactive Demo
==========================
BitNet b1.58 + Mamba-2 Hybrid Language Model

Usage:
    python demo.py                    # Interactive mode
    python demo.py --prompt "Hello"   # Single prompt
    python demo.py --benchmark        # Run benchmark
"""

import sys
import os
import time
import argparse

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.model import EFCModel
from src.data.bpe_tokenizer import RegexBPETokenizer


class AEGIS85M:
    """AEGIS-85M model wrapper for easy inference."""

    def __init__(self, model_path: str = "best_model.pt", tokenizer_path: str = "tokenizer_32k.json"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading AEGIS-85M on {self.device}...")

        # Load tokenizer
        self.tokenizer = RegexBPETokenizer.load(tokenizer_path)
        print(f"Tokenizer loaded: {self.tokenizer.vocab_size} tokens")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint.get("config", {})

        # Initialize model
        self.model = EFCModel(
            vocab_size=self.tokenizer.vocab_size,
            d_model=config.get("d_model", 768),
            n_layers=config.get("n_layers", 12),
            n_heads=config.get("n_heads", 12),
            d_state=config.get("d_state", 16),
            d_conv=config.get("d_conv", 4),
            expand=config.get("expand", 2),
            max_seq_len=512,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Model info
        self.total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded: {self.total_params:,} parameters")
        print(f"Architecture: d_model={config.get('d_model')}, n_layers={config.get('n_layers')}, n_heads={config.get('n_heads')}")
        print()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
    ) -> str:
        """Generate text from prompt."""
        # Encode
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        # Generate
        start_time = time.time()
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        elapsed = time.time() - start_time

        # Decode
        output_text = self.tokenizer.decode(output_ids[0].tolist())

        # Stats
        new_tokens = output_ids.shape[1] - len(tokens)
        tokens_per_sec = new_tokens / elapsed if elapsed > 0 else 0

        return output_text, new_tokens, tokens_per_sec

    def benchmark(self, prompts: list = None, runs: int = 3):
        """Run benchmark on sample prompts."""
        if prompts is None:
            prompts = [
                "The meaning of life is",
                "Once upon a time there was",
                "In the beginning",
                "Hello, my name is",
                "The quick brown fox",
            ]

        print("=" * 60)
        print("AEGIS-85M BENCHMARK")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Parameters: {self.total_params:,}")
        print(f"Runs per prompt: {runs}")
        print()

        total_tokens = 0
        total_time = 0

        for prompt in prompts:
            times = []
            tokens_list = []

            for _ in range(runs):
                start = time.time()
                _, new_tokens, _ = self.generate(prompt, max_new_tokens=50)
                elapsed = time.time() - start
                times.append(elapsed)
                tokens_list.append(new_tokens)

            avg_time = sum(times) / len(times)
            avg_tokens = sum(tokens_list) / len(tokens_list)
            tok_per_sec = avg_tokens / avg_time if avg_time > 0 else 0

            total_tokens += sum(tokens_list)
            total_time += sum(times)

            print(f"Prompt: \"{prompt[:30]}...\"")
            print(f"  Avg tokens: {avg_tokens:.1f}, Avg time: {avg_time:.3f}s, Tok/s: {tok_per_sec:.1f}")

        print()
        print("-" * 60)
        overall_tok_per_sec = total_tokens / total_time if total_time > 0 else 0
        print(f"Overall: {total_tokens} tokens in {total_time:.2f}s = {overall_tok_per_sec:.1f} tok/s")
        print("=" * 60)


def interactive_mode(model: AEGIS85M):
    """Run interactive chat mode."""
    print("=" * 60)
    print("AEGIS-85M Interactive Demo")
    print("=" * 60)
    print("Commands:")
    print("  /quit     - Exit")
    print("  /temp X   - Set temperature (default: 0.8)")
    print("  /tokens X - Set max tokens (default: 100)")
    print("=" * 60)
    print()

    temperature = 0.8
    max_tokens = 100

    while True:
        try:
            prompt = input(">> ").strip()

            if not prompt:
                continue

            if prompt.lower() == "/quit":
                print("Goodbye!")
                break

            if prompt.startswith("/temp "):
                try:
                    temperature = float(prompt.split()[1])
                    print(f"Temperature set to {temperature}")
                except:
                    print("Usage: /temp 0.8")
                continue

            if prompt.startswith("/tokens "):
                try:
                    max_tokens = int(prompt.split()[1])
                    print(f"Max tokens set to {max_tokens}")
                except:
                    print("Usage: /tokens 100")
                continue

            # Generate
            output, new_tokens, tok_per_sec = model.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

            print()
            print(f"<< {output}")
            print(f"   [{new_tokens} tokens, {tok_per_sec:.1f} tok/s]")
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="AEGIS-85M Demo")
    parser.add_argument("--prompt", type=str, help="Single prompt to generate from")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--model", type=str, default="best_model.pt", help="Model path")
    parser.add_argument("--tokenizer", type=str, default="tokenizer_32k.json", help="Tokenizer path")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max new tokens")
    args = parser.parse_args()

    # Load model
    model = AEGIS85M(args.model, args.tokenizer)

    if args.benchmark:
        model.benchmark()
    elif args.prompt:
        output, new_tokens, tok_per_sec = model.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Output: {output}")
        print(f"\n[{new_tokens} tokens, {tok_per_sec:.1f} tok/s]")
    else:
        interactive_mode(model)


if __name__ == "__main__":
    main()
