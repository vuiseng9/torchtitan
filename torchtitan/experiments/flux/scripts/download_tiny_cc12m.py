# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Download a tiny subset of CC12M dataset for quick testing and development.

This script downloads only a small portion of the CC12M dataset (default: 1024 samples)
from the HuggingFace streaming dataset, filters problematic samples, resizes images,
and saves them in the same format as the full clean_cc12m.py script.

Usage:
    python download_tiny_cc12m.py --output_dir /dataset/cc12m_tiny --num_samples 1024
    
This avoids downloading the full ~1TB CC12M dataset.
"""

import argparse
import os
from typing import List, Set

from datasets import Dataset, load_dataset
from torchvision.transforms import functional as F
from tqdm import tqdm


# Problematic indices from the full dataset - these are indices into the streaming dataset
# For the tiny subset, we'll skip samples that appear problematic during download
PROBLEMATIC_INDICES_FILE = os.path.join(
    os.path.dirname(__file__), "problematic_indices.txt"
)


def load_problematic_indices(filter_file: str) -> Set[int]:
    """Load problematic indices from file."""
    if not os.path.exists(filter_file):
        return set()
    with open(filter_file, "r") as f:
        indices = [line.strip() for line in f.readlines()]
        return {int(i) for i in indices if i}


def is_valid_sample(sample: dict, output_size: int) -> bool:
    """Check if a sample is valid (not problematic, sufficient resolution)."""
    try:
        img_key = "png" if "png" in sample else "jpg"
        img = sample[img_key]
        
        # Check minimum resolution
        if img.size[0] < output_size or img.size[1] < output_size:
            return False
        
        # Check for valid text
        if not sample.get("txt") or len(sample.get("txt", "").strip()) == 0:
            return False
        
        # Check image mode - we need to be able to convert to RGB
        if img.mode not in ["RGB", "RGBA", "L", "CMYK", "P"]:
            return False
            
        return True
    except Exception:
        return False


def resize_image(img, output_size: int):
    """Resize and center crop image to output_size x output_size."""
    resized_img = F.resize(
        img, output_size, interpolation=F.InterpolationMode.BICUBIC
    )
    cropped_img = F.center_crop(resized_img, (output_size, output_size))
    
    # Convert to RGB if needed
    if cropped_img.mode != "RGB":
        cropped_img = cropped_img.convert("RGB")
    
    return cropped_img


def download_tiny_cc12m(
    output_dir: str,
    num_samples: int = 1024,
    output_size: int = 256,
    seed: int = 42,
    num_workers: int = 4,
):
    """
    Download a tiny subset of CC12M from HuggingFace streaming.
    
    Args:
        output_dir: Directory to save the processed dataset
        num_samples: Number of samples to download (default: 1024)
        output_size: Output image size (default: 256x256)
        seed: Random seed for reproducibility
        num_workers: Number of workers for parallel processing
    """
    print(f"Downloading tiny CC12M subset ({num_samples} samples)...")
    print(f"Output directory: {output_dir}")
    print(f"Output size: {output_size}x{output_size}")
    
    # Load problematic indices
    problematic = load_problematic_indices(PROBLEMATIC_INDICES_FILE)
    print(f"Loaded {len(problematic)} problematic indices to skip")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load streaming dataset
    print("Loading CC12M streaming dataset from HuggingFace...")
    ds = load_dataset(
        "pixparse/cc12m-wds",
        split="train",
        streaming=True,
    )
    
    # Collect valid samples
    collected_samples = []
    sample_idx = 0
    skipped_problematic = 0
    skipped_invalid = 0
    
    print(f"Collecting {num_samples} valid samples...")
    pbar = tqdm(total=num_samples, desc="Downloading")
    
    for sample in ds:
        # Skip problematic indices
        if sample_idx in problematic:
            skipped_problematic += 1
            sample_idx += 1
            continue
        
        # Skip invalid samples
        if not is_valid_sample(sample, output_size):
            skipped_invalid += 1
            sample_idx += 1
            continue
        
        # Process the sample
        try:
            img_key = "png" if "png" in sample else "jpg"
            processed_img = resize_image(sample[img_key], output_size)
            
            collected_samples.append({
                "__key__": f"sample_{len(collected_samples):06d}",
                "png": processed_img,
                "txt": sample["txt"],
            })
            pbar.update(1)
            
            if len(collected_samples) >= num_samples:
                break
                
        except Exception as e:
            skipped_invalid += 1
        
        sample_idx += 1
    
    pbar.close()
    
    print(f"\nCollection summary:")
    print(f"  Collected: {len(collected_samples)} samples")
    print(f"  Skipped (problematic): {skipped_problematic}")
    print(f"  Skipped (invalid/error): {skipped_invalid}")
    print(f"  Total examined: {sample_idx}")
    
    if len(collected_samples) < num_samples:
        print(f"\nWarning: Only collected {len(collected_samples)} samples "
              f"(requested {num_samples})")
    
    # Create HuggingFace Dataset
    print("\nCreating dataset...")
    dataset = Dataset.from_dict({
        "__key__": [s["__key__"] for s in collected_samples],
        "png": [s["png"] for s in collected_samples],
        "txt": [s["txt"] for s in collected_samples],
    })
    
    # Save to disk
    print(f"Saving dataset to {output_dir}...")
    dataset.save_to_disk(output_dir, num_proc=num_workers)
    
    print(f"\nDone! Dataset saved to {output_dir}")
    print(f"Dataset size: {len(dataset)} samples")
    
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a tiny subset of CC12M dataset for testing"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/dataset/cc12m_tiny",
        help="Output directory for the dataset (default: /dataset/cc12m_tiny)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1024,
        help="Number of samples to download (default: 1024). "
             "Should be divisible by batch_size * num_gpus for preprocessing.",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=256,
        help="Output image size (default: 256)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for saving (default: 4)",
    )
    args = parser.parse_args()
    
    download_tiny_cc12m(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        output_size=args.output_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )
