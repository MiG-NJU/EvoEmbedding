<p align="center">
  <img src="docs/assets/icon.png" alt="EvoEmbedding" width="100%" />
</p>

<div align="center">

# EvoEmbedding

**Evolvable Embedding for Long-Context Retrieval**

[Project](https://github.com/Clare-Nie/EvoEmbedding) ·
[Dataset](https://huggingface.co/datasets/ClareNie/EvoEmbedding-Dataset) ·
[Models](https://huggingface.co/ClareNie/EvoEmbedding-4B)

</div>

---

EvoEmbedding is a memory-aware framework for long-context retrieval. Instead of encoding each text segment as an isolated static vector, EvoEmbedding maintains an evolvable latent memory and generates query-sensitive representations for retrieving information from long, temporally structured histories.

This repository provides the released code for model definition, training, inference client, and evaluation scripts. The released resources include the training dataset and model checkpoints on Hugging Face.

## Contents

- [Resources](#resources)
- [Overview](#overview)
- [Performance](#performance)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Citation](#citation)

## Resources

| Resource | Link |
| --- | --- |
| Project repository | https://github.com/Clare-Nie/EvoEmbedding |
| Training dataset | https://huggingface.co/datasets/ClareNie/EvoEmbedding-Dataset |
| Released model | https://huggingface.co/ClareNie/EvoEmbedding-4B |

## Overview

EvoEmbedding is designed for retrieval settings where the relevant evidence depends on conversation history, temporal position, or evolving user memory. The model performs two coupled operations:

- **Memory evolution**: compresses historical segments into latent memory states and updates a FIFO memory queue.
- **Representation generation**: combines latent memory with the current segment to produce contextual representations for retrieval.

<p align="center">
  <img src="docs/assets/framework.png" alt="EvoEmbedding overview" width="95%" />
</p>

Compared with static embedding models, EvoEmbedding can make retrieval decisions that are sensitive to temporal cues such as beginning, middle, and recent context.

## Performance

EvoEmbedding is evaluated on long-context retrieval and memory-oriented benchmarks, including:

- `locomo`
- `longmemeval_s`
- `personamem32`
- `PersonaMME32`
- `PersonaMME128`

<p align="center">
  <img src="docs/assets/performance.png" alt="EvoEmbedding performance" width="95%" />
</p>

<p align="center">
  <img src="docs/assets/Comparsion.png" alt="EvoEmbedding comparison" width="85%" />
</p>

## Quick Start

### Environment

```bash
pip install -r requirements.txt
```

Recommended runtime:

- Python 3.10+
- PyTorch with CUDA support
- BF16-capable GPU

### Usage

`model/client.py` provides `EvoEmbeddingClient` and `EvoRAGClient`. By default, the client loads the released EvoEmbedding checkpoint:

```python
from model.client import EvoEmbeddingClient

client = EvoEmbeddingClient(model_path="ClareNie/EvoEmbedding-4B")

messages = [
    {"role": "user", "content": "I visited Paris in April."},
    {"role": "assistant", "content": "Noted."},
    {"role": "user", "content": "Where did I travel in spring?"},
]

answer = client.send_message_raw(messages)
```

### Training

```bash
PYTHONPATH=. torchrun --nproc_per_node=8 train/train.py \
  --dataset_name ClareNie/EvoEmbedding-Dataset \
  --base_model Qwen/Qwen3-4B-Instruct-2507 \
  --output_dir ./output/evoembedding-4b
```

### Evaluation

Run a single benchmark:

```bash
PYTHONPATH=. python eval/eval.py \
  --eval_method rag \
  --model_name EvoEmbedding \
  --eval_bench locomo \
  --rag_sentence_num 16 \
  --embedding_model Qwen/Qwen3-Embedding-0.6B
```

Run the batch evaluation script:

```bash
PYTHONPATH=. bash eval/eval.sh
```

The current evaluation entrypoint keeps the following benchmarks:

- `locomo`
- `longmemeval_s`
- `personamem32`
- `PersonaMME32`
- `PersonaMME128`

## Repository Structure

```text
EvoEmbedding/
├── model/              # model implementation and client
├── train/              # training entrypoint
├── eval/               # evaluation scripts
├── docs/               # project page and visual assets
└── requirements.txt
```

## Notes

- This repository does not include benchmark data.
- The Hugging Face model repo contains inference files only.
- Evaluation scripts expect benchmark data under `./data/`.

## Citation

```bibtex
@article{nie2026evoembedding,
  title={Evolvable Embedding for Long-Context Retrieval},
  author={Nie, Chang and Fu, Chaoyou and Shan, Caifeng},
  journal={arXiv preprint},
  year={2026}
}
```

