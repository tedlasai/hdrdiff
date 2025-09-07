import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-FLF2V-14B-720P", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-FLF2V-14B-720P", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-FLF2V-14B-720P", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-FLF2V-14B-720P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern=["data/examples/wan/first_frame.jpeg", "data/examples/wan/last_frame.jpeg"]
)

# First and last frame to video
video = pipe(
    prompt="写实风格，一个女生手持枯萎的花站在花园中，镜头逐渐拉远，记录下花园的全貌。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    input_image=Image.open("data/examples/wan/first_frame.jpeg").resize((960, 960)),
    end_image=Image.open("data/examples/wan/last_frame.jpeg").resize((960, 960)),
    seed=0, tiled=True,
    height=960, width=960, num_frames=33,
    sigma_shift=16,
)
save_video(video, "video.mp4", fps=15, quality=5)
