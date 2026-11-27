import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from base import pca_lowrank_transform
from utils import load_image, registration_samples


def spatial_cosine_similarity(tensor1, tensor2, eps=1e-8, eta=0):
    similarity = F.cosine_similarity(tensor1, tensor2, dim=1, eps=eps)
    return (similarity > eta).unsqueeze(1)


def manifold_alignment(warped, fixed, confidence):
    """Squared Frobenius difference of the retained patches' cosine matrices."""
    source = F.normalize(warped.flatten(2).transpose(1, 2), dim=-1)
    target = F.normalize(fixed.flatten(2).transpose(1, 2), dim=-1)
    keep = confidence.flatten(2).transpose(1, 2).to(source.dtype)

    # SS^T - TT^T = MD^T + DM^T, where M=(S+T)/2 and D=S-T.
    # Contract in channel space to include all retained pairs without P-by-P matrices.
    midpoint = (source + target) * (keep * 0.5)
    difference = (source - target) * keep
    midpoint_t = midpoint.transpose(1, 2)
    difference_t = difference.transpose(1, 2)
    cross = midpoint_t @ difference
    squared = (midpoint_t @ midpoint) * (difference_t @ difference)
    squared = 2 * (squared.sum() + (cross * cross.transpose(1, 2)).sum())
    return squared.clamp_min(0)


def sampling_grid(flow):
    """Target-to-source sampling coordinates for a pixel-space (x, y) field."""
    height, width = flow.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    identity = torch.stack((x, y), dim=-1)
    scale = flow.new_tensor((width - 1, height - 1)).clamp_min(1)
    return 2 * (identity + flow.permute(0, 2, 3, 1)) / scale - 1


def warp(image, grid, mode="bilinear"):
    return F.grid_sample(image, grid, mode=mode, padding_mode="border", align_corners=True)


def load_features(folder, device):
    directory = folder / "dino_ebd"
    tensors = [
        torch.load(directory / f"{name}_ebd.pt", map_location=device, weights_only=True).float()
        for name in ("fixed", "moving")
    ]
    fixed, moving = (value.unsqueeze(0) if value.ndim == 3 else value for value in tensors)
    if fixed.ndim != 4 or fixed.shape != moving.shape or fixed.shape[0] != 1:
        raise ValueError(f"{folder}: expected one equal-sized fixed/moving feature pair")
    if min(fixed.shape[-2:]) < 2:
        raise ValueError(f"{folder}: the feature grid needs at least two patches per axis")

    _, channels, height, width = fixed.shape
    patches = height * width
    components = min(256, channels, 2 * patches)
    if components == channels:
        return fixed, moving

    joint = torch.cat((fixed.flatten(2), moving.flatten(2)), dim=2)[0].T
    reduced, _ = pca_lowrank_transform(joint, components)
    fixed, moving = reduced.T.reshape(1, components, 2, height, width).unbind(dim=2)
    return fixed, moving


def load_mask_pair(path, size, device):
    cache = torch.load(path, map_location=device, weights_only=True)
    masks = []
    for key in ("mask1", "mask2", "bbox"):
        value = cache[key].float()
        if key == "bbox" and value.ndim == 2:
            value = value.unsqueeze(0)
        if value.ndim == 3:
            value = value.unsqueeze(0)
        if value.ndim != 4 or value.shape[0] != 1:
            raise ValueError(f"{path}: {key} must have shape [1, K, H, W] or [K, H, W]")
        masks.append(value)

    fixed, moving, boxes = masks
    if fixed.shape[1] != moving.shape[1] or boxes.shape[1] not in (1, fixed.shape[1]):
        raise ValueError(f"{path}: mask channels must correspond to the shared prompt boxes")

    fixed = F.interpolate(fixed, size=size, mode="bilinear", align_corners=True)
    moving = F.interpolate(moving, size=size, mode="bilinear", align_corners=True)
    boxes = F.interpolate(boxes, size=size, mode="nearest-exact").bool()
    return fixed * boxes, moving * boxes, boxes


def optimize_pair(fixed, moving, masks, args):
    height, width = fixed.shape[-2:]
    field = fixed.new_zeros((1, 2, height, width), requires_grad=True)
    optimizer = torch.optim.Adam([field], lr=args.lr)
    for _ in range(args.iterations):
        optimizer.zero_grad()
        grid = sampling_grid(field)
        warped = warp(moving, grid)
        confidence = spatial_cosine_similarity(warped, fixed, eta=args.confidence_threshold)
        feature_loss = torch.linalg.vector_norm((warped - fixed) * confidence)
        smoothness = torch.linalg.vector_norm(torch.stack((
            torch.linalg.vector_norm(field[:, :, 1:, :] - field[:, :, :-1, :]),
            torch.linalg.vector_norm(field[:, :, :, 1:] - field[:, :, :, :-1]),
        )))
        loss = feature_loss + args.smooth_weight * smoothness
        if masks is not None:
            fixed_masks, moving_masks, box_support = masks
            residual = (warp(moving_masks, grid) - fixed_masks) * box_support
            semantic_loss = torch.linalg.vector_norm(residual.flatten(2), dim=-1).sum()
            loss = loss + args.semantic_weight * semantic_loss
        if args.manifold_weight:
            loss = loss + args.manifold_weight * manifold_alignment(warped, fixed, confidence)
        loss.backward()
        optimizer.step()
    return field.detach()


def main():
    parser = argparse.ArgumentParser(description="Optimize a TFSA-REG field for each cached image pair.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--smooth-weight", type=float, default=1.0)
    parser.add_argument("--semantic-weight", type=float, default=10.0, help="Use 0 for mask-free pre-registration.")
    parser.add_argument("--manifold-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0, help="Seed for joint feature PCA.")
    args = parser.parse_args()
    if args.iterations < 1 or (args.size and min(args.size) < 2):
        parser.error("Use at least one iteration and image axes of at least two pixels.")
    torch.manual_seed(args.seed)

    samples = registration_samples(args.data_root, prefix=args.split)[:args.limit]
    if not samples:
        parser.error(f"No registration samples in {args.data_root}")
    args.output.mkdir(parents=True, exist_ok=True)
    errors = []
    for folder in tqdm(samples, desc="Registering"):
        fixed, moving = load_features(folder, args.device)
        masks = None
        if args.semantic_weight:
            cache = folder / "mask.pt"
            if not cache.is_file() and "_" in folder.name:
                cache = folder.parent / folder.name.split("_", 1)[1] / "mask.pt"
            masks = load_mask_pair(cache, fixed.shape[-2:], args.device)
        field = optimize_pair(fixed, moving, masks, args)
        size = args.size or (None, None)
        with torch.no_grad():
            image = load_image(folder / "moving.jpg", args.device, *size)
            height, width = image.shape[-2:]
            feature_height, feature_width = field.shape[-2:]
            scale = field.new_tensor(((width - 1) / (feature_width - 1), (height - 1) / (feature_height - 1)))
            image_field = F.interpolate(field, (height, width), mode="bilinear", align_corners=True)
            image_field = image_field * scale.view(1, 2, 1, 1)
            registered = warp(image, sampling_grid(image_field), mode="bicubic").clamp(0, 1)
            pixels = (registered[0] * 255).round().byte().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(pixels[:, :, 0] if pixels.shape[-1] == 1 else pixels).save(args.output / f"{folder.name}.jpg")

            reference = folder / "temp.jpg"
            if reference.is_file():
                expected = load_image(reference, args.device, *size)
                if expected.shape != registered.shape:
                    raise ValueError(f"{reference}: reference shape differs from the registered image")
                errors.append(F.mse_loss(registered, expected).item())
    if errors:
        print(f"Mean reference-image MSE over {len(errors)} pairs: {sum(errors) / len(errors):.6f}")


if __name__ == "__main__":
    main()
