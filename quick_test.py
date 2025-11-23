#!/usr/bin/env python3
"""
Quick test to verify basic training works.
"""
import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.lora_model import get_model_and_tokenizer, create_lora_model, get_device
from datasets import load_dataset


class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=64):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': enc['input_ids'].squeeze(0),
        }


def main():
    print("=" * 50)
    print("Quick Training Test")
    print("=" * 50)

    device = get_device()
    print(f"Device: {device}")

    # Load minimal data
    print("\nLoading 500 samples...")
    ds = load_dataset('ag_news', split='train')
    texts = ds['text'][:500]

    # Load model
    print("Loading DistilGPT2 + LoRA...")
    base_model, tokenizer = get_model_and_tokenizer('distilgpt2')
    model = create_lora_model(base_model, 'distilgpt2', lora_r=4, lora_alpha=8)
    model.to(device)

    # Create dataloader
    dataset = SimpleTextDataset(texts, tokenizer, max_length=64)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    print(f"\nDataset size: {len(dataset)}")
    print(f"Batches: {len(loader)}")

    # Train for a few batches
    print("\nTraining for 20 batches...")
    model.train()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5
    )

    losses = []
    for i, batch in enumerate(tqdm(loader, total=min(20, len(loader)))):
        if i >= 20:
            break

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f"\nFirst loss: {losses[0]:.4f}")
    print(f"Last loss: {losses[-1]:.4f}")
    print(f"Loss decreased: {losses[-1] < losses[0]}")
    print("\nTest PASSED!" if losses[-1] < losses[0] * 1.5 else "\nTest FAILED - loss exploded")


if __name__ == '__main__':
    main()
