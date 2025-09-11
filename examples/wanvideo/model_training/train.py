import torch, os
import sys
import yaml
from pathlib import Path

sys.path.append("/data2/saikiran.tedla/hdrvideo/diff")

from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.unified_dataset import UnifiedDataset
from diffsynth.trainers.stuttgart_dataset import StuttgartDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"



class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        # Disable hub fetch — force local
        for cfg in model_configs:
            cfg.skip_download = True

        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        self.pipe.dit.require_vae_embedding = False  # Enable image VAE embeddings
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
        
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            elif extra_input == "condition_video":
                #assert num_hdr_frames is 3N + 1
                assert (len(data["video"]) - 1) % 3 == 0
                num_hdr_frames = (len(data["video"]) - 1) // 3
                inputs_shared["condition_video"] = data["video"][0:num_hdr_frames+1] #get first N+1 frames (input LDR video)
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


def load_yaml_config(args, config_path):
    with open(config_path, "r") as f:
        config_args = yaml.safe_load(f) or {}

    for k, v in config_args.items():
        setattr(args, k, v)   # always set, no checks

    return args

#[\"/data2/saikiran.tedla/hdrvideo/diff/models/train/Wan2.2-TI2V-5B_full/epoch-0.safetensors\"]"
def convert_strlist_to_jsonlist(strlist):
    if strlist is None:
        return None
    import json
    # strlist is already a list → just dump it to JSON
    return json.dumps(strlist)
    


def set_load_paths(args):
    if args.model_paths is not None:
        return args
    else:
        checkpoint_dir = Path(args.output_path) / "checkpoints"
        #find the latest checkpoint
        if checkpoint_dir.exists(): #use efa
            print("Looking for checkpoints in:", checkpoint_dir)
            checkpoint_files = list(checkpoint_dir.glob("epoch-*.safetensors"))
            if len(checkpoint_files) > 0:
                latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
                latest_checkpoint = Path(latest_checkpoint)  # ensure it's a Path object
                args.model_paths = convert_strlist_to_jsonlist([str(latest_checkpoint)])
                print(f"Resuming from checkpoint: {args.model_paths}")

                epochs_done = int(latest_checkpoint.stem.split("-")[1]) + 1
                args.epochs_done = epochs_done
                
                return args
        else: #use default weights (training from scratch)
            default_model_id_with_origin_paths = ",Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors"
            args.model_id_with_origin_paths += default_model_id_with_origin_paths
            print("No checkpoints found, training from scratch.")
            return args

if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    args = load_yaml_config(args, args.config)
    args = set_load_paths(args)

    dataset = StuttgartDataset(
        base_path=args.dataset_base_path,
        repeat=args.dataset_repeat,
        main_data_operator=StuttgartDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    model_logger = ModelLogger(
        Path(args.output_path) / "checkpoints",
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    launch_training_task(dataset, model, model_logger, args=args)
