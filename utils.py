import cv2
import numpy as np
import os
from torchmetrics.image import PeakSignalNoiseRatio
import torch

def average_frame_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    Computes the average per-frame PSNR between two HDR videos.

    Args:
        pred (torch.Tensor): Predicted video of shape [B, C, T, H, W].
        target (torch.Tensor): Ground-truth video of shape [B, C, T, H, W].
        data_range (float): Max value of the signal (1.0 if normalized).

    Returns:
        torch.Tensor: Scalar tensor with average PSNR over frames.
    """
    assert pred.shape == target.shape, "Prediction and target must have the same shape"
    B, C, T, H, W = pred.shape
    #move both tensors to cpu
    pred = pred.cpu()
    target = target.cpu()
    psnr_metric = PeakSignalNoiseRatio(data_range=data_range)

    #clamp values to [0, data_range]
    pred = torch.clamp(pred, 0.0, data_range)
    target = torch.clamp(target, 0.0, data_range)
    
    psnrs = []
    for t in range(T):
        # Extract frame t: shape [B, C, H, W]
        psnr_val = psnr_metric(pred[:, :, t], target[:, :, t])
        psnrs.append(psnr_val)

    return torch.stack(psnrs).mean()

def output_hdr_video(hdr_video, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    #write out each frame of hdr_video to path as .rad file
    N, C, H, W = hdr_video.shape
    for i in range(N):
        frame = hdr_video[i].permute(1, 2, 0).cpu().numpy()  # H, W, C
        frame = frame[:, :, ::-1]  # Convert RGB to BGR
        filename = f"{out_folder}/frame_{i:04d}.hdr"
        cv2.imwrite(filename, frame)  # Save as .hdr image

# def weight_function(video_rgb, eps=1e-6):
#     Y = 0.25 * video_rgb[:, 0] + 0.5 * video_rgb[:, 1] + 0.25 * video_rgb[:, 2]
#     w = 1.0 - 2.0 * np.abs(Y - 0.5)
#     w = np.clip(w, 0.0, None) + eps
#     w = w[:, np.newaxis, :, :].repeat(1, 3, 1, 1)  # Make it 3-channel
#     return w

def weight_function(video, eps=1e-6):
    # video: float in [0,1] (or at least roughly normalized exposures)
    # Triangular hat peaking at 0.5, zero at 0 and 1
    w = 1.0 - 2.0 * np.abs(video - 0.5)
    return np.clip(w, 0.0, None) + eps

def merge_hdr(low_exposure, normal_exposure, high_exposure, low_radiance, normal_radiance, high_radiance):
    """
    Merge low, normal, and high exposure images into an HDR image.

    Args:
        low_exposure: np.ndarray of shape (N, H, W, 3), low exposure images.
        normal_exposure: np.ndarray of shape (N, H, W, 3), normal exposure images.
        high_exposure: np.ndarray of shape (N, H, W, 3), high exposure images.
    Returns:
        np.ndarray of shape (N, H, W, 3)
    """
    w_low = weight_function(low_exposure)
    w_normal = weight_function(normal_exposure)
    w_high = weight_function(high_exposure)

    #if pixel is clipped in low_exposure, set the weight to w_low very high
    w_low[low_exposure >= 0.95] = 100
    w_high[high_exposure <= 0.05] = 100

    numerator = (w_low * low_radiance) + (w_normal * normal_radiance) + (w_high * high_radiance)
    denominator = w_low + w_normal + w_high + 1e-8  # Avoid division by zero

    hdr_image = numerator / denominator
    return hdr_image

def process_bracketed_video(video):
    #assert that video has 12 frames
    assert video.shape[0] == 12, "Video must have 12 frames for bracketed HDR processing"
    normal_exposure = video[0:4] #EV 0
    low_exposure = video[4:8] #EV -4
    high_exposure = video[8:12] #EV +4

    normal_radiance = normal_exposure
    low_radiance = low_exposure * (2 ** 4)  # EV -4
    high_radiance = high_exposure * (2 ** -4)  # EV +4

    hdr_video = merge_hdr(low_exposure, normal_exposure, high_exposure, low_radiance, normal_radiance, high_radiance)

    return hdr_video
