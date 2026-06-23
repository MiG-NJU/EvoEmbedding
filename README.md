<p align="center">
  <img src="docs/assets/icon.png" alt="EvoEmbedding" width="100%" />
</p>

<div align="center">

# EvoEmbedding

**Evolvable Embedding for Long-Context Retrieval**

[Project](https://github.com/Clare-Nie/EvoEmbedding) ·
[Dataset](https://huggingface.co/datasets/ClareNie/EvoEmbedding-Dataset) ·
[0.8B](https://huggingface.co/ClareNie/EvoEmbedding-0.8B) ·
[2B](https://huggingface.co/ClareNie/EvoEmbedding-2B) ·
[4B](https://huggingface.co/ClareNie/EvoEmbedding-4B)

</div>

---

EvoEmbedding is a memory-aware embedding framework for long-context retrieval. This repository provides the model implementation, training entrypoint, and evaluation scripts used in the release.

## Highlights

- Evolvable memory and representation generation in one model.
- Support for long-context retrieval and memory-oriented evaluation.
- Released checkpoints at 0.8B, 2B, and 4B.
- A compact codebase for training, inference, and benchmarking.

## Released Resources

| Resource | Link |
| --- | --- |
| Project repository | https://github.com/Clare-Nie/EvoEmbedding |
| Dataset | https://huggingface.co/datasets/ClareNie/EvoEmbedding-Dataset |
| EvoEmbedding-0.8B | https://huggingface.co/ClareNie/EvoEmbedding-0.8B |
| EvoEmbedding-2B | https://huggingface.co/ClareNie/EvoEmbedding-2B |
| EvoEmbedding-4B | https://huggingface.co/ClareNie/EvoEmbedding-4B |

## Repository Layout

- `model/`: EvoEmbedding model implementation and client.
- `train/train.py`: training entrypoint.
- `eval/eval.py`: benchmark evaluation entrypoint.
- `eval/eval.sh`: batch launcher for benchmark sweeps.
- `docs/`: static project page assets.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Train with the released dataset:

```bash
PYTHONPATH=. torchrun --nproc_per_node=8 train/train.py \
  --dataset_name ClareNie/EvoEmbedding-Dataset \
  --base_model Qwen/Qwen3-4B-Instruct-2507 \
  --output_dir ./output/evoembedding-4b
```

Run a single evaluation:

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

## Model Client

`model/client.py` provides:

- `EvoEmbeddingClient` / `EvoRAGClient`
- `OpenAIClient`
- `qwen3_client`

By default, `EvoEmbeddingClient` loads `ClareNie/EvoEmbedding-4B`.

## Notes

- This repository does not include the dataset itself.
- The Hugging Face model repos contain inference files only.
- The evaluation scripts expect benchmark data under `./data/`.

## Citation

```bibtex
@article{nie2026evoembedding,
  title={Evolvable Embedding for Long-Context Retrieval},
  author={Nie, Chang and Fu, Chaoyou and Shan, Caifeng},
  journal={arXiv preprint},
  year={2026}
}
```

