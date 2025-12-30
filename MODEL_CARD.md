---
language:
  - en
  - de
license: mit
library_name: pytorch
tags:
  - text-generation
  - language-model
  - bitnet
  - mamba
  - efficient
  - ternary-quantization
  - bilingual
datasets:
  - custom
pipeline_tag: text-generation
model-index:
  - name: AEGIS-85M
    results:
      - task:
          type: text-generation
        metrics:
          - name: Validation Perplexity
            type: perplexity
            value: 1.9
          - name: Validation Loss
            type: loss
            value: 0.6162
---

# AEGIS-85M

**A 85M parameter bilingual language model using BitNet b1.58 ternary quantization and Mamba-2 state space models.**

## Model Description

AEGIS-85M is an experimental language model that combines:
- **BitNet b1.58**: Ternary weight quantization (-1, 0, +1) for 32x memory compression
- **Mamba-2 SSM**: State space models for efficient O(n) sequence processing
- **Sliding Window Attention**: Local attention every 4th layer for explicit recall

### Model Architecture

| Component | Configuration |
|-----------|---------------|
| Hidden Size | 768 |
| Layers | 12 |
| Attention Heads | 12 |
| SSM State Dim | 16 |
| SSM Conv Size | 4 |
| MLP Expansion | 2 |
| Max Sequence | 512 |
| Vocabulary | 23,139 (BPE) |

### Intended Use

This model is intended for:
- Research on efficient language model architectures
- Proof-of-concept for BitNet + Mamba hybrid designs
- CPU-friendly inference experiments
- Bilingual (EN/DE) text generation

### Limitations

- Early-stage model (1,000 training steps only)
- Output coherence is limited
- Not suitable for production use
- May generate incorrect or nonsensical text

## Training

### Training Data
Mixed English/German corpus (~6M tokens) with BPE tokenization.

### Training Procedure
- **Optimizer**: AdamW
- **Learning Rate**: 3e-4 with cosine decay to 3e-5
- **Batch Size**: 32 (8 x 4 gradient accumulation)
- **Steps**: 1,000
- **Hardware**: Single GPU

### Training Metrics

| Step | Val Loss | Val PPL |
|------|----------|---------|
| 50 | 1.1567 | 3.2 |
| 200 | 0.8450 | 2.3 |
| 500 | 0.7041 | 2.0 |
| 750 | 0.6444 | 1.9 |
| 1000 | 0.6162 | 1.9 |

## Usage

```python
import torch
from src.model import EFCModel
from src.data.bpe_tokenizer import RegexBPETokenizer

# Load
tokenizer = RegexBPETokenizer.load("tokenizer_32k.json")
checkpoint = torch.load("best_model.pt", map_location="cuda")

model = EFCModel(
    vocab_size=tokenizer.vocab_size,
    d_model=768, n_layers=12, n_heads=12,
    d_state=16, d_conv=4, expand=2,
).cuda()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Generate
prompt = "Hello, my name is"
tokens = tokenizer.encode(prompt)
input_ids = torch.tensor([tokens]).cuda()

with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=50, temperature=0.8)
print(tokenizer.decode(output[0].tolist()))
```

## Environmental Impact

- **Hardware**: Single consumer GPU
- **Training Time**: ~4 hours
- **Estimated CO2**: Minimal (short training run)

## Citation

```bibtex
@misc{aegis85m2024,
  title={AEGIS-85M: BitNet b1.58 + Mamba-2 Hybrid Language Model},
  author={Hardani, Keyvan},
  year={2024},
  publisher={GitHub},
  url={https://github.com/Keyvanhardani/Aegis-85M}
}
```

## Model Card Contact

- **Author**: Keyvan Hardani
- **Website**: [keyvan.ai](https://keyvan.ai)
- **Repository**: [github.com/Keyvanhardani/EFC-Core](https://github.com/Keyvanhardani/Aegis-85M)
