import cv2
import numpy as np
import os

def output_hdr_video(hdr_video, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    #write out each frame of hdr_video to path as .rad file
    N, C, H, W = hdr_video.shape
    for i in range(N):
        frame = hdr_video[i].permute(1, 2, 0).cpu().numpy()  # H, W, C
        frame = frame[:, :, ::-1]  # Convert RGB to BGR
        print("Frame shape:", frame.shape)
        filename = f"{out_folder}/frame_{i:04d}.hdr"
        print("Mean of frame:", np.mean(frame))
        cv2.imwrite(filename, frame)  # Save as .hdr image