import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoProcessor
from trl import SFTConfig, SFTTrainer

from model.evo_embedding import EvoRAGConfig, EvoRAGModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train EvoEmbedding.")
    parser.add_argument("--dataset_name", default="ClareNie/EvoEmbedding-Dataset")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--base_model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output_dir", default="./output/evoembedding-4b")
    parser.add_argument("--num_latents", type=int, default=16)
    parser.add_argument("--buffer_capacity", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=32768 * 4)
    parser.add_argument("--chat_template", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    processor = AutoProcessor.from_pretrained(args.base_model)
    if args.chat_template:
        with open(args.chat_template, "r", encoding="utf-8") as f:
            processor.chat_template = f.read()

    config = EvoRAGConfig(
        base_model_name_or_path=args.base_model,
        num_latents=args.num_latents,
        buffer_capacity=args.buffer_capacity,
        use_internal_checkpoint=False,
        future_buffer_update_prob=0.0,
    )
    model = EvoRAGModel(config, processor)

    training_args = SFTConfig(
        model_init_kwargs={"dtype": torch.bfloat16},
        report_to="none",
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs='{"min_lr_rate": 0.1, "num_cycles": 0.5}',
        assistant_only_loss=True,
        logging_steps=1,
        gradient_checkpointing=False,
        output_dir=args.output_dir,
        save_total_limit=2,
        group_by_length=True,
        save_steps=args.save_steps,
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=processor,
    )
    trainer.train()


if __name__ == "__main__":
    main()
