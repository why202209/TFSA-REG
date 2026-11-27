import argparse
from pathlib import Path

import torch
from einops import rearrange
from PIL import Image
from torchvision.transforms import Normalize
from tqdm import tqdm

from utils import VitExtractor, load_image, registration_samples


@torch.inference_mode()
def extract_features(extractor, path, size, facet, layer, device):
    image = load_image(path, device, *size, rgb=True)
    height = extractor.get_height_patch_num(image.shape)
    width = extractor.get_width_patch_num(image.shape)
    if min(height, width) < 2:
        raise ValueError(f"{path}: increase --size or --gamma to get at least two patches per axis")

    # Keep all complete patches when the upscaled image has a partial patch at an edge.
    patch_size = extractor.get_patch_size()
    image = image[..., :height * patch_size, :width * patch_size]
    image = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)

    if facet == "tokens" and hasattr(extractor.model, "get_intermediate_layers"):
        tokens, = extractor.model.get_intermediate_layers(image, n=[layer], norm=False)
    else:
        methods = {
            "tokens": extractor.get_feature_from_input,
            "keys": extractor.get_keys_from_input,
            "queries": extractor.get_queries_from_input,
            "values": extractor.get_values_from_input,
        }
        tokens = methods[facet](image, layers=[layer])[:, 1:]
    return rearrange(tokens, "b (h w) c -> b c h w", h=height, w=width).cpu()


def main():
    parser = argparse.ArgumentParser(description="Cache pretrained DINOv2 features for each image pair.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--model", default="dinov2_vitl14", choices=(
        "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"
    ))
    parser.add_argument("--facet", default="tokens", choices=("tokens", "keys", "queries", "values"))
    parser.add_argument("--layer", type=int, help="Zero-based layer. Defaults to the ninth layer from the end.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.gamma < 1 or (args.size and min(args.size) < 2):
        parser.error("Use gamma >= 1 and image axes of at least two pixels.")

    samples = registration_samples(args.data_root, prefix=args.split)[:args.limit]
    if not samples:
        parser.error(f"No registration samples in {args.data_root}")
    extractor = VitExtractor(model_name=args.model, device=args.device, stride=14).eval()
    layers = extractor.get_n_layers()
    layer = layers - 9 if args.layer is None else args.layer
    if not 0 <= layer < layers:
        parser.error(f"The layer must be between 0 and {layers - 1}.")

    for folder in tqdm(samples, desc="Extracting DINOv2 features"):
        paths = (folder / "fixed.jpg", folder / "moving.jpg")
        size = args.size
        if size is None:
            sizes = []
            for path in paths:
                with Image.open(path) as image:
                    sizes.append((image.height, image.width))
            if sizes[0] != sizes[1]:
                raise ValueError(f"{folder}: fixed and moving sizes differ. Supply --size to resize both.")
            size = sizes[0]
        upscaled_size = tuple(axis * args.gamma for axis in size)
        features = [
            extract_features(extractor, path, upscaled_size, args.facet, layer, args.device)
            for path in paths
        ]
        directory = folder / "dino_ebd"
        directory.mkdir(exist_ok=True)
        for path, feature in zip(paths, features):
            torch.save(feature, directory / f"{path.stem}_ebd.pt")


if __name__ == "__main__":
    main()
