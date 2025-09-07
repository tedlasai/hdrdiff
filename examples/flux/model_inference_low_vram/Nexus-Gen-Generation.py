import importlib
import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig


if importlib.util.find_spec("transformers") is None:
    raise ImportError("You are using Nexus-GenV2. It depends on transformers, which is not installed. Please install it with `pip install transformers==4.49.0`.")
else:
    import transformers
    assert transformers.__version__ == "4.49.0", "Nexus-GenV2 requires transformers==4.49.0, please install it with `pip install transformers==4.49.0`."


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/Nexus-GenV2", origin_file_pattern="model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="DiffSynth-Studio/Nexus-GenV2", origin_file_pattern="generation_decoder.bin", offload_device="cpu"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors", offload_device="cpu"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/", offload_device="cpu"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors", offload_device="cpu"),
    ],
    nexus_gen_processor_config=ModelConfig(model_id="DiffSynth-Studio/Nexus-GenV2", origin_file_pattern="processor/"),
)
pipe.enable_vram_management()

prompt = "一只可爱的猫咪"
image = pipe(
    prompt=prompt, negative_prompt="",
    seed=0, cfg_scale=3, num_inference_steps=50,
    height=1024, width=1024,
)
image.save("cat.jpg")
