# AEGIS-85M: BitNet b1.58 + Mamba-2 Hybrid Language Model

A bilingual (English/German) language model built with **ternary quantization** and **state space models** for efficient CPU inference.

## Model Details

| Property | Value |
|----------|-------|
| **Parameters** | 85.8M |
| **Architecture** | BitNet b1.58 + Mamba-2 Hybrid |
| **Quantization** | Ternary (-1, 0, +1) weights |
| **Languages** | English, German |
| **Vocab Size** | 23,139 (BPE) |
| **Context Length** | 512 tokens |
| **Training Steps** | 1,000 |
| **Best Val Loss** | 0.6162 |
| **Best Val PPL** | 1.9 |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AEGIS-85M Architecture                   │
├─────────────────────────────────────────────────────────────┤
│  Embedding: vocab_size=23139, d_model=768                   │
│                                                             │
│  12x Hybrid Blocks:                                         │
│  ├── RMSNorm → SSM (Mamba-style) → Residual                │
│  ├── RMSNorm → BitLinear MLP → Residual                    │
│  └── Every 4th layer: Sliding Window Attention              │
│                                                             │
│  LM Head: d_model=768 → vocab_size=23139                   │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

- **BitLinear**: Ternary weight quantization {-1, 0, +1} with Straight-Through Estimator (STE)
- **RMSNorm**: Pre-normalization for stable training
- **Selective SSM**: Mamba-style state space model for efficient sequence mixing
- **Sliding Window Attention**: Local attention (every 4th layer) for explicit recall

## Download Model Weights

The model weights (`best_model.pt`, ~983MB) are too large for GitHub. Download options:

1. **GitHub Releases**: Check the [Releases](https://github.com/Keyvanhardani/Aegis-85M/releases) page
2. **Contact**: Request from author via [keyvan.ai](https://keyvan.ai)

After downloading, place `best_model.pt` in the repository root.

## Usage

### Quick Start

```python
import torch
from src.model import EFCModel
from src.data.bpe_tokenizer import RegexBPETokenizer

# Load tokenizer
tokenizer = RegexBPETokenizer.load("tokenizer_32k.json")

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load("best_model.pt", map_location=device)

model = EFCModel(
    vocab_size=tokenizer.vocab_size,
    d_model=768,
    n_layers=12,
    n_heads=12,
    d_state=16,
    d_conv=4,
    expand=2,
    max_seq_len=512,
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Generate text
prompt = "The meaning of life is"
tokens = tokenizer.encode(prompt)
input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

with torch.no_grad():
    output_ids = model.generate(input_ids, max_new_tokens=50, temperature=0.8)

print(tokenizer.decode(output_ids[0].tolist()))
```

### Run Demo

```bash
python demo.py
```

## Training Details

### Dataset
- Mixed English/German corpus
- ~6M tokens training data
- BPE tokenization with regex pre-tokenization (GPT-2 style)

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 8 |
| Gradient Accumulation | 4 |
| Effective Batch | 32 |
| Learning Rate | 3e-4 → 3e-5 (cosine) |
| Warmup Steps | 200 |
| Sequence Length | 128 |
| Optimizer | AdamW |

### Training Curve

```
Step    Val Loss    Val PPL
----    --------    -------
  50      1.1567      3.2
 200      0.8450      2.3
 500      0.7041      2.0
 750      0.6444      1.9
1000      0.6162      1.9  (final)
```

## Model Files

| File | Size | Description |
|------|------|-------------|
| `best_model.pt` | 983 MB | Best checkpoint (step 1000) |
| `tokenizer_32k.json` | 784 KB | BPE tokenizer vocabulary |
| `metrics.json` | 2.8 KB | Training metrics history |
| `src/` | - | Model source code |

## Limitations

- **Early Stage**: This is a proof-of-concept model trained for only 1,000 steps
- **Coherence**: Output may be inconsistent due to limited training
- **Small Dataset**: Trained on ~6M tokens (production models use 100B+)

## Future Work

1. **Extended Training**: 10K-100K steps for better coherence
2. **Larger Dataset**: FineWeb-Edu + OSCAR-DE for comprehensive coverage
3. **Fine-tuning**: Instruction tuning for chat/assistant capabilities
4. **ONNX Export**: CPU-optimized inference with ternary weight packing

## Citation

```bibtex
@misc{aegis85m2024,
  title={AEGIS-85M: BitNet b1.58 + Mamba-2 Hybrid Language Model},
  author={Keyvan Hardani},
  year={2024},
  url={https://github.com/Keyvanhardani/Aegis-85M}
}
```

## License

MIT License

## Acknowledgments

- BitNet b1.58 paper by Microsoft Research
- Mamba architecture by Albert Gu and Tri Dao
- Training infrastructure: AEGIS Supervisor System

---

**System Architect**: [Keyvan.ai](https://keyvan.ai)
