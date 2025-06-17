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
    
    This function computes dense optical flow between consecutive frames using
    the Farneback algorithm. The flow is calculated on the green channel of
    the input frames, which typically provides good contrast for motion detection.
    
    Args:
        frame (np.ndarray): The current frame in BGR format.
        previous_frame (np.ndarray): The previous frame in BGR format.
        mask (Optional[np.ndarray]): Optional mask to apply to the optical flow.
            Non-zero values in the mask indicate regions where flow should be computed.
        
    Returns:
        np.ndarray: The computed optical flow as a 2D array of shape (H, W, 2),
            where the two channels represent horizontal and vertical flow components.
    """
    # Extract green channel from both frames for optical flow computation
    previous_g_channel = previous_frame[:, :, 1]
    frame_g_channel = frame[:, :, 1]

    # Compute dense optical flow using Farneback method
    optical_flow = cv2.calcOpticalFlowFarneback(
        prev=previous_g_channel,
        next=frame_g_channel,
        flow=None,
        pyr_scale=0.1,      # Image scale (<1) to build pyramids for each image
        levels=10,          # Number of pyramid levels including initial image
        winsize=31,         # Averaging window size for polynomial expansion
        iterations=10,      # Number of iterations at each pyramid level
        poly_n=7,           # Size of pixel neighborhood for polynomial expansion
        poly_sigma=1.2,     # Standard deviation of Gaussian for polynomial expansion
        flags=0,            # Operation flags (0 for default behavior)
    )

    # Apply mask if provided
    if mask is not None:
        optical_flow[mask == 0] = 0

    return optical_flow.astype(np.float32)