#!/usr/bin/env python3
"""
Self-contained test to verify WeightedRandomSampler balances classes.
- Does NOT import training.py (avoids triggering full training)
- Uses torchvision ImageFolder directly
- Builds the sampler inline (inverse frequency weights)
- Runs a very short loop (2 epochs x 5 batches) to report batch class balance
"""

import os
from collections import Counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms


def build_weighted_sampler(dataset: ImageFolder):
    """Create WeightedRandomSampler and class weights from an ImageFolder dataset."""
    # targets are the class indices for each sample
    targets = [label for _, label in dataset.samples]

    # Count samples per class
    class_counts = Counter(targets)

    # Inverse-frequency class weights
    num_samples = len(targets)
    num_classes = len(dataset.classes)

    class_weights = {}
    for class_idx in range(num_classes):
        count = class_counts.get(class_idx, 0)
        class_weights[class_idx] = (
            (num_samples / (num_classes * count)) if count > 0 else 1.0
        )

    # Sample weights per item
    sample_weights = [class_weights[label] for _, label in dataset.samples]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler, class_counts, class_weights


def summarize_batches(dataloader: DataLoader, max_batches: int = 5):
    """Iterate a few batches and return per-batch class count summaries."""
    batch_summaries = []
    for batch_idx, (_, labels) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        counts = Counter(labels.tolist())
        batch_summaries.append(counts)
    return batch_summaries


def average_distribution(batch_summaries):
    """Compute average per-batch class counts and a simple balance ratio."""
    if not batch_summaries:
        return {}, 0.0

    all_classes = set()
    for s in batch_summaries:
        all_classes.update(s.keys())

    avg_dist = {}
    for c in sorted(all_classes):
        total = sum(s.get(c, 0) for s in batch_summaries)
        avg_dist[c] = total / len(batch_summaries)

    if not avg_dist:
        return {}, 0.0

    max_count = max(avg_dist.values())
    min_count = min(avg_dist.values())
    balance_ratio = (min_count / max_count) if max_count > 0 else 0.0
    return avg_dist, balance_ratio


def test_weighted_sampler(train_root: str, epochs: int = 2, max_batches: int = 5) -> bool:
    if not os.path.isdir(train_root):
        print(f"❌ Training folder not found: {train_root}")
        return False

    print("🧪 WeightedRandomSampler quick test (no full training)")
    print("=" * 60)
    print(f"Dataset: {train_root}")

    # Basic transform just to load images
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = ImageFolder(train_root, transform=transform)
    print(f"Classes: {dataset.classes}")
    print(f"Total samples: {len(dataset)}")

    sampler, class_counts, class_weights = build_weighted_sampler(dataset)
    print(f"Class counts: {dict(class_counts)}")
    print(f"Class weights: { {k: round(v, 4) for k, v in class_weights.items()} }")

    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=0)

    success = True
    for epoch in range(1, epochs + 1):
        batch_summaries = summarize_batches(dataloader, max_batches=max_batches)
        avg_dist, balance_ratio = average_distribution(batch_summaries)

        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  Average class counts per batch: {avg_dist}")
        print(f"  Balance ratio (min/max): {balance_ratio:.3f}")

        # Heuristic: each class should have at least ~30% of the max per-batch count
        if balance_ratio < 0.30:
            success = False

    print("\n✅ Sampler appears to balance classes well" if success else "⚠️ Sampler balance could be improved")
    return success


if __name__ == "__main__":
    train_folder = os.path.join(os.getcwd(), "usable_data", "train")
    ok = test_weighted_sampler(train_folder, epochs=2, max_batches=5)
    print("\n🎉 Test completed" if ok else "\n❌ Test completed with warnings")
