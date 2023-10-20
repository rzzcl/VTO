import argparse
import os
import itertools
import logging
import shutil

import diffusers
import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.checkpoint
import torchvision
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, AutoProcessor
from networks import ConditionGenerator, VGGLoss, load_checkpoint, save_checkpoint, make_grid
from cascade_unet.model_feature_unet import CascadeUNetModel
from dataset.dresscode import DressCodeDataset
from dataset.vitonhd import VitonHDDataset
from models.AutoencoderKL import AutoencoderKL
from models.inversion_adapter import InversionAdapter
from utils.encode_text_word_embedding import encode_text_word_embedding
from utils.image_from_pipe import generate_images_from_tryon_pipe
from utils.set_seeds import set_seed
from utils.val_metrics import compute_metrics
from vto_pipelines.tryon_pipe import StableDiffusionTryOnePipeline

logger = get_logger(__name__, log_level="INFO")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_START_METHOD"] = "thread"
torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():

    parser = argparse.ArgumentParser(description="VTO training script.")
    #data
    parser.add_argument("--dataset", type=str, required=True, choices=["dresscode", "vitonhd"], help="dataset to use")
    parser.add_argument('--dresscode_dataroot', type=str, help='DressCode dataroot')
    parser.add_argument('--vitonhd_dataroot', type=str,default="/root/data1/diffusion_virtual_try_on/dataset", help='VitonHD dataroot')
    parser.add_argument("--save_output_dir",type=str,required=True,help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--test_order", type=str, default="unpaired", choices=["unpaired", "paired"])
    parser.add_argument("--cloth_input_type", type=str, choices=["warped", "none"], default='warped',help="cloth input type. If 'warped' use the warped cloth, if none do not use the cloth as input of the unet")
    # parser.add_argument("--num_vstar", default=16, type=int, help="Number of predicted v* images to use")
    parser.add_argument("--use_clip_cloth_features", action="store_true",help="Whether to use precomputed clip cloth features是否使用clip提取的服装特征")
    
    #model
    #预训练的stable-diffusion\cascade_unet模型
    parser.add_argument("--pretrained_model_name_or_path", type=str,default="/root/data1/VTO/stable_diffuison_checkpoint",help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--cascade_checkpoint", type=str,default="",help="Path to pretrained cascade unet feature model")
    parser.add_argument("--diff_checkpoint", type=str,default="",help="Path to pretrained cascade unet feature model")


    #随机种子
    parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")

    #train
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--test_batch_size", type=int, default=16, help="Batch size (per device) for the testing dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Total number of training epochs to perform.")
    #如果提供--max_train_steps就会覆盖--num_train_epochs
    parser.add_argument("--max_train_steps",type=int,default=200001,help="Total number of training steps to perform.  If provided, overrides num_train_epochs.")
    #workers
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to use in the dataloaders.")
    parser.add_argument("--num_workers_test", type=int, default=8,help="Number of workers to use in the test dataloaders.")
    #learning rate
    parser.add_argument("--learning_rate",type=float,default=1e-5,help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--lr_scheduler",type=str,default="constant_with_warmup",help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'),)
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    #Adam优化器
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer,防止分母为0")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    #执行向后/更新传递之前要累积的更新步骤数:参数用于指定在执行一次参数更新之前要累积的梯度步数。例如，如果 gradient_accumulation_steps 的值为 4，则需要在累积了 4 个小批次的梯度后才会进行一次参数更新。
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,help="Number of updates steps to accumulate before performing a backward/update pass.")
    #是否使用渐变检查点以节省内存为代价降低向后传递速度
    # parser.add_argument(
    #     "--gradient_checkpointing",
    #     action="store_true",
    #     help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    # )
    
    #是否允许在Ampere GPU上使用TF32。可用于加快训练速度？？？？
    # parser.add_argument(
    #     "--allow_tf32",
    #     action="store_true",
    #     help=(
    #         "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
    #         " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
    #     ),
    # )

    #是否使用混合精度，一种优化模型训练效率和内存使用的技术
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default='fp16',
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    #报告
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    #分布式训练
    parser.add_argument("--local_rank", type=int, default=-1,help="For distributed training: local_rank")
    parser.add_argument("--save_checkpointing_steps",type=int,default=50000,help="How many steps do you save weights")

    parser.add_argument("--resume_from_checkpoint",type=str,default=None,help="是否要从上次保存的checkpoint开始训练")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.Xformers 是 Facebook 出品的一款加速 Transformers 的工具库,xformers 可以加速图片生成,并显著减少显存占用,")
    parser.add_argument("--uncond_fraction", type=float, default=0.2, help="Fraction of unconditioned training samples")
    
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def train(args, unet, vae,cascade_unet, optimizer, train_dataloader, lr_scheduler, test_dataloader, accelerator,noise_scheduler,val_scheduler,weight_dtype):
    total_batch_size = args.train_batch_size * accelerator.num_processes* args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    first_epoch = 0
    global_step = 0

    # Only show the progress bar once on each machine.进度条
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        cascade_unet.eval()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                #-----------------------------input--------------------------------------------------------------------------------
                # Convert images to latent space
                latents = vae.encode(batch["image"].to(weight_dtype))[0].latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Get the pose map and resize it to the same size as the latents
                pose_map = batch["pose_map"].to(accelerator.device)
                pose_map = torch.nn.functional.interpolate(pose_map, size=(pose_map.shape[2] // 8, pose_map.shape[3] // 8), mode="bilinear")

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                target = noise

                # Get the masked image 除衣服外的其他区域
                masked_image = batch["im_mask"].to(weight_dtype)
                masked_image_latents = vae.encode(masked_image)[0].latent_dist.sample() * vae.config.scaling_factor

                # Get the warped cloths latents
                if args.cloth_input_type == 'warped':
                    cloth_latents = vae.encode(batch['warped_cloth'].to(weight_dtype))[0].latent_dist.sample()
                elif args.cloth_input_type == 'none':
                    cloth_latents = None
                else:
                    raise ValueError(f"Unknown cloth input type {args.cloth_input_type}")

                if cloth_latents is not None:
                    cloth_latents = cloth_latents * vae.config.scaling_factor
                
                unet_input = torch.cat(
                        [noisy_latents, masked_image_latents, pose_map.to(weight_dtype), cloth_latents], dim=1)
                #---------------------------------------------------------------------------------------------------------------------
                #------------------------------------------cascade_unet: extract feature feature-----------------------------------------------------------------
                
                encoder_hidden_states = cascade_unet(batch["cloth"].to(accelerator.device))  #list

                #------------------------------------------get predition and conputer loss-----------------------------------------------------------------
                model_pred = unet(unet_input, timesteps, encoder_hidden_states).sample

                # loss in accelerator.autocast according to docs https://huggingface.co/docs/accelerate/v0.15.0/quicktour#mixed-precision-training
                with accelerator.autocast():
                    loss = F.mse_loss(model_pred, target, reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # Save checkpoint every checkpointing_steps steps
                if global_step % args.checkpointing_steps == 0:
                    unet.eval()                
                    if accelerator.is_main_process:
                        os.makedirs(os.path.join(args.output_dir, "checkpoint"), exist_ok=True)
                        accelerator_state_path = os.path.join(args.output_dir, "checkpoint",
                                                            f"checkpoint-{global_step}")
                        accelerator.save_state(accelerator_state_path)
                        # Unwrap the Unet
                        unwrapped_unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=True)
                        with torch.no_grad():
                            #改！
                            val_pipe = StableDiffusionTryOnePipeline(
                                # text_encoder=text_encoder,
                                vae=vae,
                                unet=unwrapped_unet,
                                cascade_unet=cascade_unet,
                                # tokenizer=tokenizer,
                                scheduler=val_scheduler,
                            ).to(accelerator.device)

                            # Extract the images  #要改！
                            with torch.cuda.amp.autocast():
                                generate_images_from_tryon_pipe(val_pipe, cascade_unet, test_dataloader,
                                                                args.output_dir, args.test_order,
                                                                f"imgs_step_{global_step}",
                                                                args.text_usage, vision_encoder, processor,
                                                                args.cloth_input_type, num_vstar=args.num_vstar)

                            # Compute the metrics
                            metrics = compute_metrics(
                                os.path.join(args.output_dir, f"imgs_step_{global_step}_{args.test_order}"),
                                args.test_order,
                                args.dataset, 'all', ['all'], args.dresscode_dataroot, args.vitonhd_dataroot)

                            print(metrics, flush=True)
                            accelerator.log(metrics, step=global_step)
                            # Save the unet
                            unet_path = os.path.join(args.output_dir, f"unet_{global_step}.pth")
                            accelerator.save(unwrapped_unet.state_dict(), unet_path)

                        del unwrapped_unet
                        del val_pipe

                    unet.train() 

            #在进度条显示损失和学习率
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

    # End of training
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    args = parse_args()

    # ---------------------------------------------------------presetting-----------------------------------------------
    # Setup accelerator.
    # 创建了一个 Accelerator 实例，其中包括了一些参数的设置，如梯度累积步数 (gradient_accumulation_steps)、混合精度训练 (mixed_precision) 和日志报告方式 (log_with)。
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    #通过 logger.info(accelerator.state, main_process_only=False) 将加速器的状态信息记录到日志中。accelerator.state 包含了加速器的相关配置和状态信息。在这里，通过 main_process_only=False 参数来确保每个进程都记录一条日志，而不仅仅是主进程。
    #接着，根据当前进程是否为主进程，分别设置了 Transformers 和 Diffusers 的日志级别。如果是主进程，则将 Transformers 的日志级别设置为 WARNING，Diffusers 的日志级别设置为 INFO；否则，将它们的日志级别都设为 ERROR。这样可以控制不同进程的日志输出级别，以减少冗余和提高可读性。
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    #--------------------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------model-------------------------------------------------------------------------
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    val_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    val_scheduler.set_timesteps(50, device=accelerator.device)

    #潜空间
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    cascade_unet = CascadeUNetModel()

    # Load cascade Checkpoint
    if not args.cascade_checkpoint == '' and os.path.exists(args.cascade_checkpoint):
        load_checkpoint(cascade_unet, args.cascade_checkpoint,args)
        print("haven load cascade chackpoint!")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    #--------------------------------------------------------------------------------------------------------------------------------
    
    #------------------------------------------Load dataset--------------------------------------------------------------------------
    print("begin loading dataset......")
    outputlist=['c_name', 'im_name', 'cloth', 'image', 'warped_cloth', 'clip_cloth_features','im_mask','pose_map']
    # Define datasets and dataloaders.
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
            order=args.test_order,
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
            order=args.test_order,
            radius=5,
            size=(512, 384),
            outputlist=tuple(outputlist)
        )
    else:
        raise NotImplementedError(f"Unknown dataset {args.dataset}")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers_test,
    )
    #-------------------------------------------------------------------------------------------------------------------------------------

    #--------------------------------------------------Prepare everything with our `accelerator`.-----------------------------------------
    # Load Checkpoint
    if not args.diff_checkpoint == '' and os.path.exists(args.diff_checkpoint):
        # load_checkpoint(unet, args.diff_checkpoint,args)
        # unwrapped_model = accelerator.unwrap_model(unet)
        unet.load_state_dict(torch.load(args.diff_checkpoint))
        print("haven load diffusion chackpoint!")
    unet, optimizer, train_dataloader, lr_scheduler, test_dataloader = accelerator.prepare(
                        unet, optimizer, train_dataloader, lr_scheduler, test_dataloader)
    vae.to(accelerator.device, dtype=weight_dtype)
    cascade_unet.to(accelerator.device, dtype=weight_dtype)

    #-------------------------------------------------------------------------------------------------------------------------------------

    #---------------------------------------------------set log wandb---------------------------------------------------------------------
    if accelerator.is_main_process:
        accelerator.init_trackers("Diff_vto", config=vars(args),
                                init_kwargs={"wandb": {"name": os.path.basename(args.save_output_dir)}})
    if args.report_to == 'wandb':
        wandb_tracker = accelerator.get_tracker("wandb")
        wandb_tracker.name = os.path.basename(args.save_output_dir)

    #---------------------------------------------------load checkpoint |  train  |   save checkpoint--------------------------------------------------------------------
    
    train(args, unet, vae,cascade_unet, optimizer, train_dataloader, lr_scheduler, test_dataloader, accelerator,noise_scheduler,val_scheduler,weight_dtype)
    # save_checkpoint(unet, os.path.join(args.checkpoint_dir, args.name, 'diff_model_final.pth'),args)

if __name__ == "__main__":
    main()