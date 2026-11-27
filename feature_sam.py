import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from utils import registration_samples


def sample_boxes(height, width, count, rng):
    for _ in range(count):
        box_width = rng.randint(max(1, width // 4), max(1, width // 2))
        box_height = rng.randint(max(1, height // 4), max(1, height // 2))
        left = rng.randint(0, width - box_width)
        top = rng.randint(0, height - box_height)
        yield left, top, left + box_width, top + box_height


def mask_for_box(processor, state, box, support, prompt):
    processor.reset_all_prompts(state)
    if prompt:
        processor.set_text_prompt(prompt, state)
    output = processor.add_geometric_prompt(box, label=True, state=state)
    # Each prompt index has one channel. Merge detected instances within its box.
    return output["masks"].flatten(0, 1).any(dim=0).cpu() & support


def segment_pair(processor, fixed, moving, boxes, prompt):
    width, height = fixed.size
    fixed_state = processor.set_image(fixed)
    moving_state = processor.set_image(moving)
    fixed_masks, moving_masks, supports = [], [], []
    for left, top, right, bottom in boxes:
        box = [(left + right) / (2 * width), (top + bottom) / (2 * height),
               (right - left) / width, (bottom - top) / height]
        support = torch.zeros((height, width), dtype=torch.bool)
        support[top:bottom, left:right] = True
        fixed_mask = mask_for_box(processor, fixed_state, box, support, prompt)
        moving_mask = mask_for_box(processor, moving_state, box, support, prompt)
        # A box supplies mask supervision only when both images have a detection.
        if not fixed_mask.any() or not moving_mask.any():
            support.zero_()
            fixed_mask.zero_()
            moving_mask.zero_()
        fixed_masks.append(fixed_mask)
        moving_masks.append(moving_mask)
        supports.append(support)
    return {
        "mask1": torch.stack(fixed_masks).unsqueeze(0),
        "mask2": torch.stack(moving_masks).unsqueeze(0),
        "bbox": torch.stack(supports).unsqueeze(0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cache SAM3 masks for roughly aligned image pairs using shared prompt boxes."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--boxes", type=int, default=15)
    parser.add_argument("--prompt", default="building, road, river, parkland")
    parser.add_argument("--sam-confidence", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=Path, help="Local SAM3 weights. Defaults to the official Hugging Face checkpoint.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.boxes < 1 or (args.size and min(args.size) < 2):
        parser.error("Use at least one box and image axes of at least two pixels.")

    samples = registration_samples(args.data_root, prefix=args.split)[:args.limit]
    if not samples:
        parser.error(f"No registration samples in {args.data_root}")
    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ModuleNotFoundError as exc:
        parser.error(f"{exc}. Install SAM3 with the environment specified by the linked SAM3 repository.")

    checkpoint_path = str(args.checkpoint) if args.checkpoint is not None else None
    model = build_sam3_image_model(device=args.device, checkpoint_path=checkpoint_path)
    processor = Sam3Processor(model, device=args.device, confidence_threshold=args.sam_confidence)
    for folder in tqdm(samples, desc="Generating SAM3 masks"):
        with Image.open(folder / "fixed.jpg") as image:
            fixed = image.convert("RGB")
        with Image.open(folder / "moving.jpg") as image:
            moving = image.convert("RGB")
        if args.size is not None:
            fixed = fixed.resize(tuple(reversed(args.size)), Image.Resampling.LANCZOS)
            moving = moving.resize(fixed.size, Image.Resampling.LANCZOS)
        if fixed.size != moving.size:
            raise ValueError(f"{folder}: fixed and moving sizes differ. Supply --size to resize both.")
        rng = random.Random(f"{args.seed}:{folder.name}")
        boxes = sample_boxes(fixed.height, fixed.width, args.boxes, rng)
        torch.save(segment_pair(processor, fixed, moving, boxes, args.prompt), folder / "mask.pt")


if __name__ == "__main__":
    main()
