# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified for animal pose visualization (AP10k/APT36k format).
# Original: https://github.com/kijai/ComfyUI-WanAnimatePreprocess

import os
import cv2
import math
import numpy as np
from typing import Dict, List
from .pose2d_utils import AAPoseMeta


# ==================== AP10k / APT36k Animal Keypoint Definitions ====================
# 0: L_Eye, 1: R_Eye, 2: Nose, 3: Neck, 4: Root_of_tail
# 5: L_Shoulder, 6: L_Elbow, 7: L_F_Paw
# 8: R_Shoulder, 9: R_Elbow, 10: R_F_Paw
# 11: L_Hip, 12: L_Knee, 13: L_B_Paw
# 14: R_Hip, 15: R_Knee, 16: R_B_Paw

# Skeleton connections for AP10k/APT36k (0-indexed)
ANIMAL_LIMB_SEQ = [
    [2, 3],              # Nose to Neck
    [3, 4],              # Neck to Tail (spine)
    [3, 5], [5, 6], [6, 7],      # Left front leg
    [3, 8], [8, 9], [9, 10],     # Right front leg
    [4, 11], [11, 12], [12, 13], # Left back leg
    [4, 14], [14, 15], [15, 16], # Right back leg
]

ANIMAL_HEAD_LIMBS = [
    [2, 0], [2, 1],     # Nose to eyes
]

# Colors for each limb segment
ANIMAL_LIMB_COLORS = [
    [255, 0, 0],      # Nose-Neck (red)
    [255, 85, 0],     # Neck-Tail (orange)
    [255, 170, 0],    # Neck-L_Shoulder
    [255, 255, 0],    # L_Shoulder-L_Elbow
    [170, 255, 0],    # L_Elbow-L_F_Paw
    [85, 255, 0],     # Neck-R_Shoulder
    [0, 255, 0],      # R_Shoulder-R_Elbow
    [0, 255, 85],     # R_Elbow-R_F_Paw
    [0, 255, 170],    # Tail-L_Hip
    [0, 255, 255],    # L_Hip-L_Knee
    [0, 170, 255],    # L_Knee-L_B_Paw
    [0, 85, 255],     # Tail-R_Hip
    [0, 0, 255],      # R_Hip-R_Knee
    [85, 0, 255],     # R_Knee-R_B_Paw
]

ANIMAL_HEAD_COLORS = [
    [170, 0, 255],    # Nose-L_Eye
    [255, 0, 255],    # Nose-R_Eye
]

# Keypoint colors (one per keypoint)
ANIMAL_KP_COLORS = [
    [170, 0, 255],    # L_Eye
    [255, 0, 255],    # R_Eye
    [255, 0, 0],      # Nose
    [255, 85, 0],     # Neck
    [255, 170, 0],    # Tail
    [255, 255, 0],    # L_Shoulder
    [170, 255, 0],    # L_Elbow
    [85, 255, 0],     # L_F_Paw
    [0, 255, 0],      # R_Shoulder
    [0, 255, 85],     # R_Elbow
    [0, 255, 170],    # R_F_Paw
    [0, 255, 255],    # L_Hip
    [0, 170, 255],    # L_Knee
    [0, 85, 255],     # L_B_Paw
    [0, 0, 255],      # R_Hip
    [85, 0, 255],     # R_Knee
    [170, 0, 255],    # R_B_Paw
]


# ==================== Core Drawing Functions ====================

def draw_animal_pose(
    img,
    kp2ds,
    threshold=0.5,
    stick_width_norm=200,
    draw_head=True,
    dataset='ap10k'
):
    """Draw animal pose skeleton on image (AP10k or APT36k format - 17 keypoints).

    Args:
        img (np.ndarray): Input image (H, W, 3)
        kp2ds (np.ndarray): Keypoints array of shape (17, 3) - [x, y, confidence]
        threshold (float): Confidence threshold for drawing
        stick_width_norm (int): Divisor for stick width calculation
        draw_head (bool): Whether to draw head keypoints (eyes, nose)
        dataset (str): 'ap10k' or 'apt36k' (both have same skeleton)

    Returns:
        np.ndarray: Image with drawn pose
    """
    H, W, C = img.shape
    stickwidth = max(int(min(H, W) / stick_width_norm), 1)

    # Build limb list
    limb_seq = list(ANIMAL_LIMB_SEQ)
    colors = list(ANIMAL_LIMB_COLORS)
    if draw_head:
        limb_seq.extend(ANIMAL_HEAD_LIMBS)
        colors.extend(ANIMAL_HEAD_COLORS)

    # Draw limbs
    for (k1_idx, k2_idx), color in zip(limb_seq, colors):
        if k1_idx >= len(kp2ds) or k2_idx >= len(kp2ds):
            continue

        keypoint1 = kp2ds[k1_idx]
        keypoint2 = kp2ds[k2_idx]

        if keypoint1[2] < threshold or keypoint2[2] < threshold:
            continue

        Y = np.array([keypoint1[0], keypoint2[0]])
        X = np.array([keypoint1[1], keypoint2[1]])
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly(
            (int(mY), int(mX)),
            (int(length / 2), stickwidth),
            int(angle), 0, 360, 1
        )
        cv2.fillConvexPoly(img, polygon, [int(float(c) * 0.6) for c in color])

    # Draw keypoints
    for idx, keypoint in enumerate(kp2ds):
        if keypoint[2] < threshold:
            continue
        if not draw_head and idx in [0, 1]:  # Skip eyes if not drawing head
            continue

        x, y = keypoint[0], keypoint[1]
        kp_color = ANIMAL_KP_COLORS[idx] if idx < len(ANIMAL_KP_COLORS) else (0, 0, 255)
        cv2.circle(img, (int(x), int(y)), stickwidth, kp_color, thickness=-1)

    return img


def draw_animal_pose_new(
    img,
    kp2ds,
    threshold=0.5,
    draw_head=True,
    body_stick_width=-1,
    data_to_json=None,
    idx=-1,
):
    """Draw animal pose with configurable stick width.

    Args:
        img (np.ndarray): Input image (H, W, 3)
        kp2ds (np.ndarray): Keypoints array (17, 3) - [x, y, confidence]
        threshold (float): Confidence threshold
        draw_head (bool): Whether to draw head keypoints
        body_stick_width (int): Stick width (-1 for auto)
        data_to_json (list|None): Optional list to append JSON data
        idx (int): Frame index for JSON data

    Returns:
        np.ndarray: Image with drawn pose
    """
    H, W, C = img.shape
    kp2ds = kp2ds.copy()

    if not draw_head:
        kp2ds[[0, 1], 2] = 0  # Zero out eye confidence

    if body_stick_width == -1:
        stickwidth = max(int(min(H, W) / 200) - 1, 1)
    else:
        stickwidth = body_stick_width

    # Build limb list
    limb_seq = list(ANIMAL_LIMB_SEQ)
    colors = list(ANIMAL_LIMB_COLORS)
    if draw_head:
        limb_seq.extend(ANIMAL_HEAD_LIMBS)
        colors.extend(ANIMAL_HEAD_COLORS)

    # Draw limbs
    for (k1_idx, k2_idx), color in zip(limb_seq, colors):
        if k1_idx >= len(kp2ds) or k2_idx >= len(kp2ds):
            continue

        keypoint1 = kp2ds[k1_idx]
        keypoint2 = kp2ds[k2_idx]

        if keypoint1[-1] < threshold or keypoint2[-1] < threshold:
            continue

        Y = np.array([keypoint1[0], keypoint2[0]])
        X = np.array([keypoint1[1], keypoint2[1]])
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly(
            (int(mY), int(mX)),
            (int(length / 2), stickwidth),
            int(angle), 0, 360, 1
        )
        cv2.fillConvexPoly(img, polygon, [int(float(c) * 0.6) for c in color])

    # Draw keypoints
    for kp_idx, keypoint in enumerate(kp2ds):
        if keypoint[-1] < threshold:
            continue
        x, y = keypoint[0], keypoint[1]
        kp_color = ANIMAL_KP_COLORS[kp_idx] if kp_idx < len(ANIMAL_KP_COLORS) else (0, 0, 255)
        cv2.circle(img, (int(x), int(y)), stickwidth, kp_color, thickness=-1)

    # Normalize for JSON output
    kp2ds[:, 0] /= W
    kp2ds[:, 1] /= H

    if data_to_json is not None:
        entry = {
            "image_id": "frame_{:05d}.jpg".format((len(data_to_json) + 1) if idx == -1 else (idx + 1)),
            "height": H,
            "width": W,
            "category_id": 1,
            "keypoints_body": kp2ds.tolist(),
        }
        if idx == -1:
            data_to_json.append(entry)
        else:
            data_to_json[idx] = entry

    return img


# ==================== Meta-based Drawing Wrappers ====================

def draw_animal_pose_by_meta(img, meta, threshold=0.5, stick_width_norm=200, draw_head=True, dataset='ap10k', body_stick_width=-1):
    """Draw animal pose from AAPoseMeta object.

    Args:
        img (np.ndarray): Input canvas/image (H, W, 3)
        meta (AAPoseMeta): Animal pose metadata with kps_body and kps_body_p
        threshold (float): Confidence threshold
        stick_width_norm (int): Stick width normalization (ignored if body_stick_width != -1)
        draw_head (bool): Whether to draw head keypoints
        dataset (str): 'ap10k' or 'apt36k'
        body_stick_width (int): Explicit stick width (-1 for auto)

    Returns:
        np.ndarray: Image with drawn animal pose
    """
    kp2ds = np.concatenate([meta.kps_body, meta.kps_body_p[:, None]], axis=1)

    if body_stick_width != -1:
        return draw_animal_pose_new(
            img, kp2ds, threshold,
            draw_head=draw_head,
            body_stick_width=body_stick_width,
        )
    else:
        return draw_animal_pose(
            img, kp2ds, threshold,
            stick_width_norm=stick_width_norm,
            draw_head=draw_head,
            dataset=dataset,
        )


# Keep backward compatibility aliases
def draw_aapose_by_meta_new(img, meta, threshold=0.5, stickwidth_type='v2', body_stick_width=-1, draw_hand=False, draw_head=True, hand_stick_width=4):
    """Backward compatibility wrapper - redirects to animal pose drawing."""
    return draw_animal_pose_by_meta(
        img, meta, threshold,
        draw_head=draw_head,
        body_stick_width=body_stick_width,
    )


# ==================== Utility Drawing Functions ====================

def draw_ellipse_by_2kp(img, keypoint1, keypoint2, color, threshold=0.6):
    """Draw an ellipse between two keypoints."""
    H, W, C = img.shape
    stickwidth = max(int(min(H, W) / 200), 1)

    if keypoint1[-1] < threshold or keypoint2[-1] < threshold:
        return img

    Y = np.array([keypoint1[0], keypoint2[0]])
    X = np.array([keypoint1[1], keypoint2[1]])
    mX = np.mean(X)
    mY = np.mean(Y)
    length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
    angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
    polygon = cv2.ellipse2Poly(
        (int(mY), int(mX)),
        (int(length / 2), stickwidth),
        int(angle), 0, 360, 1
    )
    cv2.fillConvexPoly(img, polygon, [int(float(c) * 0.6) for c in color])
    return img


def draw_bbox(img, bbox, color=(255, 0, 0)):
    """Draw a bounding box on an image."""
    img = load_image(img)
    bbox = [int(bbox_tmp) for bbox_tmp in bbox]
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    return img


def draw_kp2ds(img, kp2ds, threshold=0, color=(255, 0, 0), skeleton=None, reverse=False):
    """Draw keypoints on an image with optional skeleton connections."""
    img = load_image(img, reverse)

    if skeleton is not None and skeleton == "ap10k":
        skeleton_list = ANIMAL_LIMB_SEQ + ANIMAL_HEAD_LIMBS
        color_list = ANIMAL_LIMB_COLORS + ANIMAL_HEAD_COLORS
        for _idx, _skeleton in enumerate(skeleton_list):
            for i in range(len(_skeleton) - 1):
                cv2.line(
                    img,
                    (int(kp2ds[_skeleton[i], 0]), int(kp2ds[_skeleton[i], 1])),
                    (int(kp2ds[_skeleton[i + 1], 0]), int(kp2ds[_skeleton[i + 1], 1])),
                    color_list[_idx % len(color_list)],
                    3,
                )

    for _idx, kp2d in enumerate(kp2ds):
        if kp2d[2] > threshold:
            cv2.circle(img, (int(kp2d[0]), int(kp2d[1])), 3, color, -1)

    return img


def load_image(img, reverse=False):
    """Load image from path or return as-is."""
    if type(img) == str:
        img = cv2.imread(img)
    if reverse:
        img = img.astype(np.float32)
        img = img[:, :, ::-1]
        img = img.astype(np.uint8)
    return img


# ==================== Skeleton Drawing from Meta Dict ====================

def draw_skeleten(meta):
    """Draw animal skeleton from a meta dictionary with normalized keypoints.

    Args:
        meta (dict): Must contain 'keypoints_body', 'height', 'width'

    Returns:
        np.ndarray: Pose image
    """
    kps = []
    for i, kp in enumerate(meta["keypoints_body"]):
        if kp is None:
            kps.append([0, 0, 0])
        else:
            kps.append([*kp, 1] if len(kp) == 2 else list(kp))
    kps = np.array(kps)

    kps[:, 0] *= meta["width"]
    kps[:, 1] *= meta["height"]
    pose_img = np.zeros([meta["height"], meta["width"], 3], dtype=np.uint8)

    pose_img = draw_animal_pose(pose_img, kps, draw_head=True)
    return pose_img


# ==================== Trajectory Visualization ====================

def draw_traj(metas: List[AAPoseMeta], threshold=0.6):
    """Draw keypoint trajectories across frames for animal pose.

    Args:
        metas: List of AAPoseMeta objects across frames
        threshold: Confidence threshold

    Returns:
        list: List of trajectory images per frame
    """
    import random

    colors = ANIMAL_LIMB_COLORS + ANIMAL_HEAD_COLORS

    # Animal skeleton limb sequence (1-indexed for compatibility)
    limbSeq = [
        [3, 4],              # Nose-Neck
        [4, 5],              # Neck-Tail
        [4, 6], [6, 7], [7, 8],      # Left front leg
        [4, 9], [9, 10], [10, 11],   # Right front leg
        [5, 12], [12, 13], [13, 14], # Left back leg
        [5, 15], [15, 16], [16, 17], # Right back leg
    ]

    kp_body = np.array([meta.kps_body for meta in metas])
    kp_body_p = np.array([meta.kps_body_p for meta in metas])

    new_limbSeq = []
    key_point_list = []
    for _idx, (k1_index, k2_index) in enumerate(limbSeq):
        vis = (kp_body_p[:, k1_index - 1] > threshold) * (kp_body_p[:, k2_index - 1] > threshold) * 1
        if vis.sum() * 1.0 / vis.shape[0] > 0.4:
            new_limbSeq.append([k1_index, k2_index])

    for _idx, (k1_index, k2_index) in enumerate(limbSeq):
        keypoint1 = kp_body[:, k1_index - 1]
        keypoint2 = kp_body[:, k2_index - 1]
        interleave = random.randint(4, 7)
        randind = random.randint(0, interleave - 1)

        Y = np.array([keypoint1[:, 0], keypoint2[:, 0]])
        X = np.array([keypoint1[:, 1], keypoint2[:, 1]])

        vis = (keypoint1[:, -1] > threshold if keypoint1.shape[-1] > 2 else np.ones(len(keypoint1))) * \
              (keypoint2[:, -1] > threshold if keypoint2.shape[-1] > 2 else np.ones(len(keypoint2)))

        t = randind / interleave
        x = (1 - t) * Y[0, :] + t * Y[1, :]
        y = (1 - t) * X[0, :] + t * X[1, :]

        x = x.astype(int)
        y = y.astype(int)

        new_array = np.array([x, y, vis]).T
        key_point_list.append(new_array)

    if len(key_point_list) == 0:
        # Fallback: use individual keypoints
        for kp_idx in range(min(kp_body.shape[1], 5)):
            kp = kp_body[:, kp_idx, :]
            kp_p = kp_body_p[:, kp_idx]
            arr = np.column_stack([kp[:, 0].astype(int), kp[:, 1].astype(int), (kp_p > threshold).astype(int)])
            key_point_list.append(arr)

    key_points_list = np.stack(key_point_list)
    num_points = len(key_points_list)
    sample_colors = random.sample(colors, min(num_points, len(colors)))
    while len(sample_colors) < num_points:
        sample_colors.extend(random.sample(colors, min(num_points - len(sample_colors), len(colors))))

    stickwidth = max(int(min(metas[0].width, metas[0].height) / 150), 2)

    image_list_ori = []
    for i in range(key_points_list.shape[-2]):
        _image_vis = np.zeros((metas[0].width, metas[0].height, 3))
        points = key_points_list[:, i, :]
        for point_idx, point in enumerate(points):
            x, y, vis = point
            if vis == 1:
                cv2.circle(_image_vis, (x, y), stickwidth, sample_colors[point_idx], thickness=-1)
        image_list_ori.append(_image_vis)

    return image_list_ori
