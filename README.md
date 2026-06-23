# EvoEmbedding

Code release for EvoEmbedding, including the model implementation, training entrypoint, and evaluation scripts.

## Resources

- Dataset: https://huggingface.co/datasets/ClareNie/EvoEmbedding-Dataset
- Model: https://huggingface.co/ClareNie/EvoEmbedding-4B

## Structure

- `model/`: EvoEmbedding model definition and inference/evaluation client.
- `train/train.py`: training entrypoint.
- `eval/`: evaluation launcher and benchmark scripts.

## Training

```bash
PYTHONPATH=. torchrun --nproc_per_node=8 train/train.py \
  --dataset_name ClareNie/EvoEmbedding-Dataset \
  --output_dir ./output/evoembedding-4b
```

## Evaluation

```bash
PYTHONPATH=. bash eval/eval.sh
```

By default, `model/client.py` loads `ClareNie/EvoEmbedding-4B`.
