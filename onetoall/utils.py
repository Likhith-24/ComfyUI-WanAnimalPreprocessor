# https://github.com/ssj9596/One-to-All-Animation
# Modified for animal pose (AP10k/APT36k - 17 keypoints, no hands/face).

import cv2
import numpy as np
import math
import copy

eps = 0.01

# AP10k keypoint groups
# Head: L_Eye(0), R_Eye(1), Nose(2)
# Spine: Neck(3), Tail(4)
# Front legs: L_Shoulder(5), L_Elbow(6), L_F_Paw(7), R_Shoulder(8), R_Elbow(9), R_F_Paw(10)
# Back legs: L_Hip(11), L_Knee(12), L_B_Paw(13), R_Hip(14), R_Knee(15), R_B_Paw(16)

DROP_HEAD_POINTS = {0, 1, 2}
DROP_FRONT_LEG_POINTS = {5, 6, 7, 8, 9, 10}
DROP_BACK_LEG_POINTS = {11, 12, 13, 14, 15, 16}


def scale_and_translate_pose(tgt_pose, ref_pose, conf_th=0.9, return_ratio=False):
    """Scale and translate target pose to match reference pose.

    Uses animal keypoint indices:
    - Shoulders: L_Shoulder(5), R_Shoulder(8)
    - Hips: L_Hip(11), R_Hip(14)
    - Anchor: Neck(3)
    """
    aligned_pose = copy.deepcopy(tgt_pose)
    th = 1e-6
    ref_kpt = ref_pose['bodies']['candidate'].astype(np.float32)
    tgt_kpt = aligned_pose['bodies']['candidate'].astype(np.float32)

    ref_sc = ref_pose['bodies'].get('score', np.ones((1, ref_kpt.shape[0]))).astype(np.float32).reshape(-1)
    tgt_sc = tgt_pose['bodies'].get('score', np.ones((1, tgt_kpt.shape[0]))).astype(np.float32).reshape(-1)

    # AP10k: L_Shoulder=5, R_Shoulder=8
    ref_shoulder_valid = (ref_sc[5] >= conf_th) and (ref_sc[8] >= conf_th)
    tgt_shoulder_valid = (tgt_sc[5] >= conf_th) and (tgt_sc[8] >= conf_th)
    shoulder_ok = ref_shoulder_valid and tgt_shoulder_valid

    # AP10k: L_Hip=11, R_Hip=14
    ref_hip_valid = (ref_sc[11] >= conf_th) and (ref_sc[14] >= conf_th)
    tgt_hip_valid = (tgt_sc[11] >= conf_th) and (tgt_sc[14] >= conf_th)
    hip_ok = ref_hip_valid and tgt_hip_valid

    if shoulder_ok and hip_ok:
        ref_shoulder_w = abs(ref_kpt[5, 0] - ref_kpt[8, 0])
        tgt_shoulder_w = abs(tgt_kpt[5, 0] - tgt_kpt[8, 0])
        x_ratio = ref_shoulder_w / tgt_shoulder_w if tgt_shoulder_w > th else 1.0

        ref_torso_h = abs(np.mean(ref_kpt[[11, 14], 1]) - np.mean(ref_kpt[[5, 8], 1]))
        tgt_torso_h = abs(np.mean(tgt_kpt[[11, 14], 1]) - np.mean(tgt_kpt[[5, 8], 1]))
        y_ratio = ref_torso_h / tgt_torso_h if tgt_torso_h > th else 1.0
        scale_ratio = (x_ratio + y_ratio) / 2

    elif shoulder_ok:
        ref_sh_dist = np.linalg.norm(ref_kpt[5] - ref_kpt[8])
        tgt_sh_dist = np.linalg.norm(tgt_kpt[5] - tgt_kpt[8])
        scale_ratio = ref_sh_dist / tgt_sh_dist if tgt_sh_dist > th else 1.0

    else:
        # Fallback: Neck-Tail spine
        ref_spine = np.linalg.norm(ref_kpt[3] - ref_kpt[4])
        tgt_spine = np.linalg.norm(tgt_kpt[3] - tgt_kpt[4])
        scale_ratio = ref_spine / tgt_spine if tgt_spine > th else 1.0

    if return_ratio:
        return scale_ratio

    # Scale around Neck (index 3)
    anchor_idx = 3
    anchor_pt = tgt_kpt[anchor_idx].copy()

    def scale(arr):
        if arr is not None and arr.size > 0:
            arr[..., 0] = anchor_pt[0] + (arr[..., 0] - anchor_pt[0]) * scale_ratio
            arr[..., 1] = anchor_pt[1] + (arr[..., 1] - anchor_pt[1]) * scale_ratio

    scale(tgt_kpt)

    offset = ref_kpt[anchor_idx] - tgt_kpt[anchor_idx]

    def translate(arr):
        if arr is not None and arr.size > 0:
            arr += offset

    translate(tgt_kpt)
    aligned_pose['bodies']['candidate'] = tgt_kpt

    return aligned_pose, shoulder_ok, hip_ok


def warp_ref_to_pose(tgt_img, ref_pose, tgt_pose, bg_val=(0, 0, 0), conf_th=0.9, align_center=False):
    """Warp reference image to match target pose using affine transform."""
    H, W = tgt_img.shape[:2]
    img_tgt_pose = draw_pose_aligned(tgt_pose, H, W)

    tgt_kpt = tgt_pose['bodies']['candidate'].astype(np.float32)
    ref_kpt = ref_pose['bodies']['candidate'].astype(np.float32)

    scale_ratio = scale_and_translate_pose(tgt_pose, ref_pose, conf_th=conf_th, return_ratio=True)

    # Anchor on Neck (index 3)
    anchor_idx = 3
    x0 = tgt_kpt[anchor_idx][0] * W
    y0 = tgt_kpt[anchor_idx][1] * H

    ref_x = ref_kpt[anchor_idx][0] * W if not align_center else W / 2
    ref_y = ref_kpt[anchor_idx][1] * H

    dx = ref_x - x0
    dy = ref_y - y0

    M = np.array([[scale_ratio, 0, (1 - scale_ratio) * x0 + dx],
                  [0, scale_ratio, (1 - scale_ratio) * y0 + dy]],
                 dtype=np.float32)

    img_warp = cv2.warpAffine(tgt_img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=bg_val)
    img_tgt_pose_warp = cv2.warpAffine(img_tgt_pose, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=bg_val)
    zeros = np.zeros((H, W), dtype=np.uint8)
    mask_warp = cv2.warpAffine(zeros, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=255)

    return img_warp, img_tgt_pose_warp, mask_warp


def hsv_to_rgb(hsv):
    """Convert HSV to RGB color."""
    hsv = np.asarray(hsv, dtype=np.float32)
    in_shape = hsv.shape
    hsv = hsv.reshape(-1, 3)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    i = i % 6

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    rgb = np.zeros_like(hsv)
    rgb[i == 0] = np.stack([v[i == 0], t[i == 0], p[i == 0]], axis=1)
    rgb[i == 1] = np.stack([q[i == 1], v[i == 1], p[i == 1]], axis=1)
    rgb[i == 2] = np.stack([p[i == 2], v[i == 2], t[i == 2]], axis=1)
    rgb[i == 3] = np.stack([p[i == 3], q[i == 3], v[i == 3]], axis=1)
    rgb[i == 4] = np.stack([t[i == 4], p[i == 4], v[i == 4]], axis=1)
    rgb[i == 5] = np.stack([v[i == 5], p[i == 5], q[i == 5]], axis=1)

    gray_mask = s == 0
    rgb[gray_mask] = np.stack([v[gray_mask]] * 3, axis=1)
    return (rgb.reshape(in_shape) * 255)


def get_stickwidth(W, H, stickwidth=4):
    """Calculate stick width based on image resolution."""
    maxdim = max(W, H)
    if maxdim < 512:
        ratio = 1.0
    elif maxdim < 1080:
        ratio = 1.5
    elif maxdim < 2160:
        ratio = 2.0
    elif maxdim < 3240:
        ratio = 2.5
    elif maxdim < 4320:
        ratio = 3.5
    else:
        ratio = 4.0
    return int(stickwidth * ratio)


def alpha_blend_color(color, alpha):
    return [int(c * alpha) for c in color]


def draw_bodypose_aligned(canvas, candidate, subset, score, plan=None):
    """Draw animal body pose skeleton.

    AP10k skeleton connections (1-indexed for compatibility with original code):
    """
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)
    stickwidth = get_stickwidth(W, H, stickwidth=3)

    # AP10k skeleton (1-indexed)
    limbSeq = [
        [3, 4],                              # Nose-Neck
        [4, 5],                              # Neck-Tail (spine)
        [4, 6], [6, 7], [7, 8],             # Left front leg
        [4, 9], [9, 10], [10, 11],          # Right front leg
        [5, 12], [12, 13], [13, 14],        # Left back leg
        [5, 15], [15, 16], [16, 17],        # Right back leg
        [3, 1], [3, 2],                     # Nose to eyes
    ]

    colors = [
        [255, 0, 0],      # Nose-Neck
        [255, 85, 0],     # Neck-Tail
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
        [170, 0, 255],    # Nose-L_Eye
        [255, 0, 255],    # Nose-R_Eye
    ]

    HIDE_JOINTS = set()
    if plan:
        if plan["mode"] == "drop_point":
            HIDE_JOINTS.add(plan["point_idx"])
        elif plan["mode"] == "drop_region":
            HIDE_JOINTS |= set(plan["points"])

    # Draw limbs
    for i, idx_pair in enumerate(limbSeq):
        if any(j in HIDE_JOINTS for j in idx_pair):
            continue

        for n in range(len(subset)):
            index = subset[n][np.array(idx_pair) - 1]
            conf = score[n][np.array(idx_pair) - 1]
            if -1 in index:
                continue

            alpha = max(conf[0] * conf[1], 0) if conf[0] > 0 and conf[1] > 0 else 0
            if conf[0] == 0 or conf[1] == 0:
                alpha = 0

            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)

            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)),
                                       (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            color = colors[i % len(colors)]
            cv2.fillConvexPoly(canvas, polygon, alpha_blend_color(color, alpha))

    canvas = (canvas * 0.6).astype(np.uint8)

    # Draw keypoints
    for i in range(17):
        if i in HIDE_JOINTS:
            continue
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            conf = score[n][i]

            alpha = 0 if conf == -2 else max(conf, 0)
            x = int(x * W)
            y = int(y * H)
            color = colors[i % len(colors)]
            cv2.circle(canvas, (x, y), stickwidth, alpha_blend_color(color, alpha), thickness=-1)

    return canvas


def draw_pose_aligned(pose, H, W, ref_w=2160, without_face=False, pose_plan=None, head_strength="full"):
    """Draw aligned animal pose image.

    Args:
        pose: dwpose-like dict with 'bodies' key
        H, W: Output image dimensions
        ref_w: Reference width for scaling
        without_face: Not used for animals (kept for API compatibility)
        pose_plan: Optional plan for hiding joints
        head_strength: "full", "weak", or "none" - controls head keypoint visibility

    Returns:
        np.ndarray: Pose image (H, W, 3)
    """
    bodies = pose['bodies']
    candidate = bodies['candidate']
    subset = bodies['subset']
    body_score = bodies['score'].copy()

    # Head control: AP10k head indices are 0 (L_Eye), 1 (R_Eye), 2 (Nose)
    if head_strength == "weak":
        target_joints = [0, 1, 2]
        body_score[:, target_joints] = -2
    elif head_strength == "none":
        target_joints = [0, 1, 2]
        body_score[:, target_joints] = 0

    sz = min(H, W)
    sr = (ref_w / sz) if sz != ref_w else 1
    canvas = np.zeros(shape=(int(H * sr), int(W * sr), 3), dtype=np.uint8)

    canvas = draw_bodypose_aligned(canvas, candidate, subset, score=body_score, plan=pose_plan)

    return cv2.resize(canvas, (W, H))
