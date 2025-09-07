accelerate launch --config_file examples/qwen_image/model_training/full/accelerate_config.yaml examples/qwen_image/model_training/train.py \
  --dataset_base_path data/example_image_dataset \
  --dataset_metadata_path data/example_image_dataset/metadata_blockwise_controlnet_inpaint.csv \
  --data_file_keys "image,blockwise_controlnet_image,blockwise_controlnet_inpaint_mask" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors,DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint:model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.blockwise_controlnet.models.0." \
  --output_path "./models/train/Qwen-Image-Blockwise-ControlNet-Inpaint_full" \
  --trainable_models "blockwise_controlnet" \
  --extra_inputs "blockwise_controlnet_image,blockwise_controlnet_inpaint_mask" \
  --use_gradient_checkpointing \
  --find_unused_parameters

# If you want to pre-train a Inpaint Blockwise ControlNet from scratch,
# please run the following script to first generate the initialized model weights file,
# and then start training with a high learning rate (1e-3).

# python examples/qwen_image/model_training/scripts/Qwen-Image-Blockwise-ControlNet-Inpaint-Initialize.py

# accelerate launch --config_file examples/qwen_image/model_training/full/accelerate_config.yaml examples/qwen_image/model_training/train.py \
#   --dataset_base_path data/example_image_dataset \
#   --dataset_metadata_path data/example_image_dataset/metadata_blockwise_controlnet_inpaint.csv \
#   --data_file_keys "image,blockwise_controlnet_image,blockwise_controlnet_inpaint_mask" \
#   --max_pixels 1048576 \
#   --dataset_repeat 50 \
#   --model_id_with_origin_paths "Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
#   --model_paths '["models/blockwise_controlnet_inpaint.safetensors"]' \
#   --learning_rate 1e-3 \
#   --num_epochs 2 \
#   --remove_prefix_in_ckpt "pipe.blockwise_controlnet.models.0." \
#   --output_path "./models/train/Qwen-Image-Blockwise-ControlNet-Inpaint_full" \
#   --trainable_models "blockwise_controlnet" \
#   --extra_inputs "blockwise_controlnet_image,blockwise_controlnet_inpaint_mask" \
#   --use_gradient_checkpointing \
#   --find_unused_parameters
