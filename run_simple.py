import torch
import sys
sys.path.append("/data2/saikiran.tedla/hdrvideo/DiffSynth-Studio-main/diffsynth")

from dotenv import load_dotenv
load_dotenv()

from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        # ModelConfig(path="/data2/saikiran.tedla/hdrvideo/diff/models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu", skip_download=True),
        # ModelConfig(path="/data2/saikiran.tedla/hdrvideo/diff/models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu", skip_download=True),
        # ModelConfig(path="/data2/saikiran.tedla/hdrvideo/diff/models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu", skip_download=True),
       
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu", skip_download=True),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu", skip_download=True),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu", skip_download=True),
  
    ],
    use_usp=False
)
pipe.enable_vram_management()

video = pipe(
    prompt="A documentary photography style scene: a lively puppy rapidly running on green grass. The puppy has brown-yellow fur, upright ears, and looks focused and joyful. Sunlight shines on its body, making the fur appear soft and shiny. The background is an open field with occasional wildflowers, and bright blue sky and clouds in the distance. Strong sense of perspective captures the motion of the puppy and the vitality of the surrounding grass. Mid-shot side-moving view.",
    negative_prompt="static, blurry details, subtitles, style, artwork, image, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, malformed limbs, fused fingers, still frame, messy background, three legs, crowded background people, walking backwards",
    seed=0,
    tiled = False,
    # num_frames = 81,
    # num_inference_steps = 10,
    height = 704,
    width = 1280,
)
save_video(video, "video1.mp4", fps=15, quality=5)