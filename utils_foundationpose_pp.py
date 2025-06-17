import numpy as np
import torch
from scipy.spatial.transform import Rotation
import time
from contextlib import contextmanager
import cv2
from PIL import Image
from typing import List

def get_6d_pose_arr_from_mat(pose):
    if torch.is_tensor(pose):
        is_batched = pose.ndim == 3
        if is_batched:
            pose_np = pose[0].cpu().numpy()
        else:
            pose_np = pose.cpu().numpy()
    else:
        pose_np = pose

    xyz = pose_np[:3, 3]
    rotation_matrix = pose_np[:3, :3]
    euler_angles = Rotation.from_matrix(rotation_matrix).as_euler('xyz', degrees=False)
    return np.r_[xyz, euler_angles]

def adjust_pose_to_image_point(
        ob_in_cam: torch.Tensor,
        K: torch.Tensor,
        x: float = -1.,
        y: float = -1.,
) -> torch.Tensor:
    """
    Adjusts the 6D pose(s) so that the projection matches the given 2D coordinate (x, y).

    Parameters:
    - ob_in_cam: Original 6D pose(s) as [4,4] or [B,4,4] tensor.
    - K: Camera intrinsic matrix (3x3 tensor).
    - x, y: Desired 2D coordinates on the image plane.

    Returns:
    - ob_in_cam_new: Adjusted pose(s) in same shape as input (tensor).
    """
    device = ob_in_cam.device
    dtype = ob_in_cam.dtype

    is_batched = ob_in_cam.ndim == 3
    if not is_batched:
        ob_in_cam = ob_in_cam.unsqueeze(0)  # [1, 4, 4]

    B = ob_in_cam.shape[0]
    ob_in_cam_new = torch.eye(4, device=device, dtype=dtype).repeat(B, 1, 1)

    for i in range(B):
        R = ob_in_cam[i, :3, :3]
        t = ob_in_cam[i, :3, 3]

        tx, ty = get_pose_xy_from_image_point(ob_in_cam[i], K, x, y)
        t_new = torch.tensor([tx, ty, t[2]], device=device, dtype=dtype)

        ob_in_cam_new[i, :3, :3] = R
        ob_in_cam_new[i, :3, 3] = t_new

    return ob_in_cam_new if is_batched else ob_in_cam_new[0]

def get_pose_xy_from_image_point(
        ob_in_cam: torch.Tensor, 
        K: torch.Tensor, 
        x: float = -1., 
        y: float = -1.,
) -> tuple:
    """
    Computes new (tx, ty) in camera space such that the projection matches image point (x, y).

    Parameters:
    - ob_in_cam: 4x4 pose tensor.
    - K: 3x3 intrinsic matrix tensor.
    - x, y: Desired image coordinates.

    Returns:
    - tx, ty: New x/y in camera coordinate system.
    """

    is_batched = ob_in_cam.ndim == 3
    if is_batched:
        ob_in_cam_new = ob_in_cam[0].cpu()  # [1, 4, 4]
    else:
        ob_in_cam_new = ob_in_cam.cpu()

    if x == -1. or y == -1.:
        return x, y
    
    t = ob_in_cam_new[:3, 3]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    tz = t[2]

    tx = (x - cx) * tz / fx
    ty = (y - cy) * tz / fy

    return tx, ty

def get_mat_from_6d_pose_arr(pose_arr):
    # 提取位移 (xyz)
    xyz = pose_arr[:3]
    
    # 提取欧拉角
    euler_angles = pose_arr[3:]
    
    # 从欧拉角生成旋转矩阵
    rotation = Rotation.from_euler('xyz', euler_angles, degrees=False)
    rotation_matrix = rotation.as_matrix()
    
    # 创建 4x4 变换矩阵
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = xyz
    
    return transformation_matrix


@contextmanager
def timing_block(label: str):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"{label}: {elapsed_time:.4f} seconds")

def visualize_mask(
        image: np.ndarray, 
        mask: np.ndarray,
        save_path: str
):
    # Ensure mask is in 'RGBA' mode
    mask_rgba = Image.fromarray(mask.astype(np.uint8)).convert('RGBA')

    # Create an alpha mask where the non-zero regions are semi-transparent
    alpha = 128  # Semi-transparent
    mask_data = mask_rgba.getdata()
    new_data = []
    for item in mask_data:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            # Background pixel, make it fully transparent
            new_data.append((0, 0, 0, 0))
        else:
            # Mask pixel, set desired transparency
            new_data.append((item[0], item[1], item[2], alpha))
    mask_rgba.putdata(new_data)

    # Convert the original image to 'RGBA'
    image_rgba = Image.fromarray(image.astype(np.uint8)).convert('RGBA')

    # Overlay the mask onto the image
    overlaid_image = Image.alpha_composite(image_rgba, mask_rgba)

    # Convert back to 'RGB' mode if you don't need transparency in the saved image
    overlaid_image = overlaid_image.convert('RGB')

    overlaid_image.save(save_path)


def visualize_bbox(
        image: np.ndarray, 
        bbox: List[int], 
        save_path: str
):
    """
    Visualize the bounding box on the image and save it to the result path.
    """
    if image is None:
        return  # If the image can't be read, skip it

    if bbox is None or bbox[0] == -1:
        return  # Skip invalid bounding boxes

    # Unpack the bounding box coordinates (x, y, w, h)
    x, y, w, h = bbox
    top_left = (int(x), int(y))
    bottom_right = (int(x + w), int(y + h))

    # Draw the bounding box on the image
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    # Save the image to the visualization path
    cv2.imwrite(save_path, image)