import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().parents[2].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import torch.utils.checkpoint
import torch.utils.checkpoint
import torchvision
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers.utils import check_min_version
from transformers import CLIPVisionModelWithProjection, AutoProcessor, CLIPProcessor
import pickle

from src.dataset.dresscode import DressCodeDataset
from src.dataset.vitonhd import VitonHDDataset
from tqdm import tqdm

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_START_METHOD"] = "thread"


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset", type=str, required=True, choices=["dresscode", "vitonhd"], help="dataset to use")
    parser.add_argument('--dresscode_dataroot', type=str, help='DressCode dataroot')
    parser.add_argument('--vitonhd_dataroot', type=str, help='VitonHD dataroot')

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-inpainting",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (per device) for the testing dataloader.")


    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of workers for the testing dataloader.")

    args = parser.parse_args()

    return args


@torch.no_grad()
def main():
    args = parse_args()

    # Check if the dataset dataroot is provided
    if args.dataset == "vitonhd" and args.vitonhd_dataroot is None:
        raise ValueError("VitonHD dataroot must be provided")
    if args.dataset == "dresscode" and args.dresscode_dataroot is None:
        raise ValueError("DressCode dataroot must be provided")

    accelerator = Accelerator()
    device = accelerator.device

    # Get the vision encoder and the processor
    if args.pretrained_model_name_or_path == "runwayml/stable-diffusion-inpainting":
        vision_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    elif args.pretrained_model_name_or_path == "stabilityai/stable-diffusion-2-inpainting":
        vision_encoder = CLIPVisionModelWithProjection.from_pretrained("/data1/ladi-vton/clip-vit-h-14")
        processor = AutoProcessor.from_pretrained("/data1/ladi-vton/clip-vit-h-14")
    else:
        raise ValueError(f"Unknown pretrained model name or path: {args.pretrained_model_name_or_path}")
    vision_encoder.requires_grad_(False)

    vision_encoder = vision_encoder.to(device)
    outputlist = ['cloth', 'c_name']

    # Get the dataset
    if args.dataset == "dresscode":
        train_dataset = DressCodeDataset(
            dataroot_path=args.dresscode_dataroot,
            phase='train',
            order='paired',
            radius=5,
            category=['dresses', 'upper_body', 'lower_body'],
            size=(512, 384),
            outputlist=tuple(outputlist)
        )

        test_dataset = DressCodeDataset(
            dataroot_path=args.dresscode_dataroot,
            phase='test',
            order='paired',
            radius=5,
            category=['dresses', 'upper_body', 'lower_body'],
            size=(512, 384),
            outputlist=tuple(outputlist)
        )
    elif args.dataset == "vitonhd":
        train_dataset = VitonHDDataset(
            dataroot_path=args.vitonhd_dataroot,
            phase='train',
            order='paired',
            radius=5,
            size=(512, 384),
            outputlist=tuple(outputlist)
        )

        test_dataset = VitonHDDataset(
            dataroot_path=args.vitonhd_dataroot,
            phase='test',
            order='paired',
            radius=5,
            size=(512, 384),
            outputlist=tuple(outputlist)
        )
    else:
        raise NotImplementedError(f"Unknown dataset: {args.dataset}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Extract the CLIP features for the clothes in the dataset and save them to disk.
    save_cloth_features(args.dataset, processor, train_loader, vision_encoder, "train")
    save_cloth_features(args.dataset, processor, test_loader, vision_encoder, "test")


def save_cloth_features(dataset: str, processor: CLIPProcessor, loader: torch.utils.data.DataLoader,
                        vision_encoder: CLIPVisionModelWithProjection, split: str):
    """
    Extract the CLIP features for the clothes in the dataset and save them to disk.
    """
    last_hidden_state_list = []
    cloth_names = []
    for batch in tqdm(loader):
        names = batch["c_name"]
        with torch.cuda.amp.autocast():
            input_image = torchvision.transforms.functional.resize((batch["cloth"] + 1) / 2, (224, 224),
                                                                   antialias=True).clamp(0, 1)
            processed_images = processor(images=input_image, return_tensors="pt")
            visual_features = vision_encoder(processed_images.pixel_values.to(vision_encoder.device))

            last_hidden_state_list.append(visual_features.last_hidden_state.cpu().half())
            cloth_names.extend(names)

    save_dir = PROJECT_ROOT / 'data' / 'clip_cloth_embeddings' / dataset
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.cat(last_hidden_state_list, dim=0), save_dir / f"{split}_last_hidden_state_features.pt")
    #在加载特征向量时，可以使用 torch.load() 函数从 .pt 文件中读取并还原为相应的张量对象。
    #总结起来，.pt 文件是 PyTorch 中用于保存和加载序列化对象（如张量、模型等）的常见文件格式，它提供了一种方便的方式来存储和传输 PyTorch 对象的数据。
    with open(os.path.join(save_dir / f"{split}_features_names.pkl"), "wb") as f:
        pickle.dump(cloth_names, f)


if __name__ == '__main__':
    main()
