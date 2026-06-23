# EvoEmbedding

EvoEmbedding is a retrieval- and memory-oriented embedding project built on top of Qwen-based models. This repository contains the model code, training entrypoint, and evaluation scripts.

## Links

- Project: https://github.com/Clare-Nie/EvoEmbedding
- Dataset: https://huggingface.co/datasets/ClareNie/EvoEmbedding-Dataset
- Model 0.8B: https://huggingface.co/ClareNie/EvoEmbedding-0.8B
- Model 2B: https://huggingface.co/ClareNie/EvoEmbedding-2B
- Model 4B: https://huggingface.co/ClareNie/EvoEmbedding-4B

## What is in this repo

- `model/`: EvoEmbedding model implementation and client.
- `train/train.py`: training entrypoint.
- `eval/eval.py`: benchmark evaluation entrypoint.
- `eval/eval.sh`: batch launcher for benchmark sweeps.
- `docs/`: static project page assets.

## Installation

```bash
pip install -r requirements.txt
```

Recommended environment:

- Python 3.10+
- PyTorch with BF16 support
- CUDA-capable GPU

## Training

The default training script loads the published dataset and a Qwen base model:

```bash
PYTHONPATH=. torchrun --nproc_per_node=8 train/train.py \
  --dataset_name ClareNie/EvoEmbedding-Dataset \
  --base_model Qwen/Qwen3-4B-Instruct-2507 \
  --output_dir ./output/evoembedding-4b
```

Key arguments:

- `--dataset_name`: Hugging Face dataset name
- `--dataset_split`: dataset split, default `train`
- `--base_model`: base model used for training
- `--num_latents`: number of latent memory slots
- `--buffer_capacity`: memory buffer size
- `--max_length`: sequence length during training
- `--chat_template`: optional custom chat template file

## Evaluation

The evaluation stack is centered on `eval/eval.py`. `eval/eval.sh` is a batch launcher that schedules multiple benchmark runs.

Single run:

```bash
PYTHONPATH=. python eval/eval.py \
  --eval_method rag \
  --model_name EvoEmbedding \
  --eval_bench locomo \
  --rag_sentence_num 16 \
  --embedding_model Qwen/Qwen3-Embedding-0.6B
```

Batch run:

```bash
PYTHONPATH=. bash eval/eval.sh
```

Supported benchmarks include:

- `locomo`
- `longmemeval_s`
- `membencheval`
- `personamem32`
- `personamem128`
- `PersonaMME32`
- `PersonaMME128`
- `PersonaRAGBench`

## Model Client

`model/client.py` exposes:

- `EvoEmbeddingClient` / `EvoRAGClient`
- `OpenAIClient`
- `qwen3_client`

By default, `EvoEmbeddingClient` loads:

- `ClareNie/EvoEmbedding-4B`

## Repository layout

```text
EvoEmbedding/
├── model/
├── train/
├── eval/
├── docs/
└── requirements.txt
```

## Notes

- This repo does not include the dataset itself.
- The Hugging Face model repos contain inference files only.
- The evaluation scripts expect the benchmark data under `./data/`.

