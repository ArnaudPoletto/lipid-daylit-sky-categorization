import cv2
import numpy as np
from typing import Optional

def get_optical_flow(
    frame: np.ndarray,
    previous_frame: np.ndarray,
    mask: Optional[np.ndarray],
) -> np.ndarray:
    """
    Calculate the optical flow between two frames using Farneback method.
    
    Args:
        frame (np.ndarray): The current frame in BGR format.
        previous_frame (np.ndarray): The previous frame in BGR format.
        mask (Optional[np.ndarray]): Optional mask to apply to the optical flow.
        
    Returns:
        np.ndarray: The computed optical flow as a 2D array of shape (H, W, 2).
    """
    previous_g_channel = previous_frame[:, :, 1]
    frame_g_channel = frame[:, :, 1]

    optical_flow = cv2.calcOpticalFlowFarneback(
        prev=previous_g_channel,
        next=frame_g_channel,
        flow=None,
        pyr_scale=0.1,
        levels=10,
        winsize=31,
        iterations=10,
        poly_n=7,
        poly_sigma=1.2,
        flags=0,
    )

    if mask is not None:
        optical_flow[mask == 0] = 0

    return optical_flow.astype(np.float32)