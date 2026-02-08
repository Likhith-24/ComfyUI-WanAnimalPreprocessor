# https://github.com/ssj9596/One-to-All-Animation
# Modified for animal pose (AP10k/APT36k - 17 keypoints, no hands/face).

import numpy as np
import copy
from ..retarget_pose import get_retarget_pose


# ==================== Convert animal pose meta to dwpose-like format ====================

def aaposemeta_to_dwpose(meta):
    """Convert an animal pose meta dict to dwpose-like format.

    Animal poses have only body keypoints (17 points), no hands or face.

    Args:
        meta (dict): Must contain 'keypoints_body' array of shape (N, 3) [x_norm, y_norm, conf]

    Returns:
        dict: dwpose-like format with bodies, hands=None, faces=None
    """
    kps_body = np.array(meta['keypoints_body'])

    candidate_body = kps_body[:, :2]
    score_body = kps_body[:, 2]
    subset_body = np.arange(len(candidate_body), dtype=float)
    subset_body[score_body <= 0] = -1

    bodies = {
        "candidate": candidate_body,
        "subset": np.expand_dims(subset_body, axis=0),
        "score": np.expand_dims(score_body, axis=0),
    }

    dwpose_format = {
        "bodies": bodies,
        "hands": None,
        "hands_score": None,
        "faces": None,
        "faces_score": None,
    }
    return dwpose_format


def aaposemeta_obj_to_dwpose(pose_meta):
    """Convert an AAPoseMeta object into dwpose-like format.

    Animal version - body keypoints only, no hands or face.

    Args:
        pose_meta (AAPoseMeta): Animal pose metadata object

    Returns:
        dict: dwpose-like format with bodies, hands=None, faces=None
    """
    w = pose_meta.width
    h = pose_meta.height

    def safe(arr, like_shape):
        if arr is None:
            return np.zeros(like_shape, dtype=np.float32)
        arr_np = np.array(arr, dtype=np.float32)
        arr_np = np.nan_to_num(arr_np, nan=0.0)
        return arr_np

    kps_body = safe(pose_meta.kps_body, (17, 2))
    candidate_body = kps_body / np.array([w, h])
    score_body = safe(pose_meta.kps_body_p, (candidate_body.shape[0],))
    subset_body = np.arange(len(candidate_body), dtype=float)
    subset_body[score_body <= 0] = -1

    bodies = {
        "candidate": candidate_body,
        "subset": np.expand_dims(subset_body, axis=0),
        "score": np.expand_dims(score_body, axis=0),
    }

    dwpose_format = {
        "bodies": bodies,
        "hands": None,
        "hands_score": None,
        "faces": None,
        "faces_score": None,
    }
    return dwpose_format


# ==================== Scale and Translate Pose ====================

def scale_and_translate_pose(tgt_pose, ref_pose, conf_th=0.9, return_ratio=False):
    """Scale and translate target pose to match reference pose dimensions.

    Uses animal-specific anchor points for alignment:
    - Shoulders: L_Shoulder(5), R_Shoulder(8)
    - Hips: L_Hip(11), R_Hip(14)
    - Neck(3) as anchor point

    Args:
        tgt_pose (dict): Target pose in dwpose format
        ref_pose (dict): Reference pose in dwpose format
        conf_th (float): Confidence threshold for keypoint validity
        return_ratio (bool): If True, return only the scale ratio

    Returns:
        If return_ratio: float scale_ratio
        Otherwise: (aligned_pose, shoulder_ok, hip_ok)
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
        # Use shoulder width and torso height for scaling
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
        # Fallback: use Neck-Tail spine length
        ref_spine = np.linalg.norm(ref_kpt[3] - ref_kpt[4])
        tgt_spine = np.linalg.norm(tgt_kpt[3] - tgt_kpt[4])
        scale_ratio = ref_spine / tgt_spine if tgt_spine > th else 1.0

    if return_ratio:
        return scale_ratio

    # Scale around Neck (index 3) as anchor
    anchor_idx = 3
    anchor_pt = tgt_kpt[anchor_idx].copy()

    def scale(arr):
        if arr is not None and arr.size > 0:
            arr[..., 0] = anchor_pt[0] + (arr[..., 0] - anchor_pt[0]) * scale_ratio
            arr[..., 1] = anchor_pt[1] + (arr[..., 1] - anchor_pt[1]) * scale_ratio

    scale(tgt_kpt)

    # Translate to match reference anchor
    offset = ref_kpt[anchor_idx] - tgt_kpt[anchor_idx]

    def translate(arr):
        if arr is not None and arr.size > 0:
            arr += offset

    translate(tgt_kpt)
    aligned_pose['bodies']['candidate'] = tgt_kpt

    return aligned_pose, shoulder_ok, hip_ok


# ==================== Compute Bone Ratio (Animal) ====================

def compute_ratios_animal(ref_scores, source_scores, ref_pts, src_pts, conf_th=0.9):
    """Compute bone length ratios between reference and source animal poses.

    AP10k skeleton (0-indexed):
    Spine: Nose(2)-Neck(3), Neck(3)-Tail(4)
    Left front: Neck(3)-L_Shoulder(5), L_Shoulder(5)-L_Elbow(6), L_Elbow(6)-L_F_Paw(7)
    Right front: Neck(3)-R_Shoulder(8), R_Shoulder(8)-R_Elbow(9), R_Elbow(9)-R_F_Paw(10)
    Left back: Tail(4)-L_Hip(11), L_Hip(11)-L_Knee(12), L_Knee(12)-L_B_Paw(13)
    Right back: Tail(4)-R_Hip(14), R_Hip(14)-R_Knee(15), R_Knee(15)-R_B_Paw(16)
    Head: Nose(2)-L_Eye(0), Nose(2)-R_Eye(1)

    Returns:
        dict: Mapping of (src_idx, dst_idx) -> ratio
    """
    th = 1e-6

    def keypoint_valid(idx):
        return ref_scores[0, idx] >= conf_th and source_scores[0, idx] >= conf_th

    def safe_ratio(p1, p2):
        len_ref = np.linalg.norm(ref_pts[p1] - ref_pts[p2])
        len_src = np.linalg.norm(src_pts[p1] - src_pts[p2])
        return len_ref / len_src if len_src > th else 1.0

    # All bones in the animal skeleton
    bone_pairs = [
        (2, 3), (3, 4),                        # Spine
        (3, 5), (5, 6), (6, 7),                # Left front leg
        (3, 8), (8, 9), (9, 10),               # Right front leg
        (4, 11), (11, 12), (12, 13),           # Left back leg
        (4, 14), (14, 15), (15, 16),           # Right back leg
        (2, 0), (2, 1),                         # Head (eyes)
    ]

    ratios = {p: 1.0 for p in bone_pairs}

    # Compute ratios for all valid bones
    for p1, p2 in bone_pairs:
        if keypoint_valid(p1) and keypoint_valid(p2):
            ratios[(p1, p2)] = safe_ratio(p1, p2)

    # Symmetrize left/right limbs
    symmetric_pairs = [
        ((3, 5), (3, 8)),       # Neck→shoulders
        ((5, 6), (8, 9)),       # Upper front legs
        ((6, 7), (9, 10)),      # Lower front legs
        ((4, 11), (4, 14)),     # Tail→hips
        ((11, 12), (14, 15)),   # Upper back legs
        ((12, 13), (15, 16)),   # Lower back legs
        ((2, 0), (2, 1)),       # Eyes
    ]

    for left_key, right_key in symmetric_pairs:
        left_val = ratios.get(left_key, 1.0)
        right_val = ratios.get(right_key, 1.0)
        avg_val = (left_val + right_val) / 2.0
        ratios[left_key] = avg_val
        ratios[right_key] = avg_val

    return ratios


# ==================== Align to Reference (Retarget) ====================

def align_to_reference(ref_pose_meta, tpl_pose_metas, tpl_dwposes, anchor_idx=None):
    """Align template poses to reference pose using retargeting.

    Animal version - no face alignment needed.

    Args:
        ref_pose_meta: Reference pose meta dict
        tpl_pose_metas: Template pose meta dicts
        tpl_dwposes: Template poses in dwpose format
        anchor_idx: Index of the anchor frame

    Returns:
        list: Retargeted poses in dwpose format
    """
    best_idx = anchor_idx if anchor_idx is not None else 0
    tpl_pose_meta_best = tpl_pose_metas[best_idx]

    tpl_retarget_pose_metas = get_retarget_pose(
        tpl_pose_meta_best,
        ref_pose_meta,
        tpl_pose_metas,
        None, None,
    )

    retarget_dwposes = [aaposemeta_obj_to_dwpose(pm) for pm in tpl_retarget_pose_metas]
    return retarget_dwposes


# ==================== Align to Pose (Stepwise Bone Scaling) ====================

def align_to_pose(ref_dwpose, tpl_dwposes, anchor_idx=None, conf_th=0.9):
    """Align template poses to reference by stepwise bone scaling.

    Animal version - scales bones according to reference proportions.

    Args:
        ref_dwpose: Reference pose in dwpose format
        tpl_dwposes: List of template poses in dwpose format
        anchor_idx: Anchor frame index
        conf_th: Confidence threshold

    Returns:
        list: Aligned poses in dwpose format
    """
    detected_poses = copy.deepcopy(tpl_dwposes)

    best_idx = anchor_idx if anchor_idx is not None else 0
    best_pose = tpl_dwposes[best_idx]
    ref_pose_scaled, _, _ = scale_and_translate_pose(ref_dwpose, best_pose, conf_th=conf_th)

    ref_candidate = ref_pose_scaled['bodies']['candidate'].astype(np.float32)
    ref_scores = ref_pose_scaled['bodies']['score'].astype(np.float32)

    source_candidate = best_pose['bodies']['candidate'].astype(np.float32)
    source_scores = best_pose['bodies']['score'].astype(np.float32)

    ratios = compute_ratios_animal(ref_scores, source_scores, ref_candidate, source_candidate, conf_th=conf_th)

    # Apply bone scaling to each frame
    for pose in detected_poses:
        candidate = pose['bodies']['candidate']

        # Define bone chains with parent→children propagation
        # Each entry: (bone_pair, end_joint, joints_to_move_with_end)
        bone_chain = [
            # Spine: Nose-Neck
            ((2, 3), 2, [0, 1]),  # Move nose and eyes when Nose-Neck changes

            # Left front leg
            ((3, 5), 5, [6, 7]),         # Neck→L_Shoulder, propagate to elbow+paw
            ((5, 6), 6, [7]),            # L_Shoulder→L_Elbow, propagate to paw
            ((6, 7), 7, []),             # L_Elbow→L_F_Paw

            # Right front leg
            ((3, 8), 8, [9, 10]),        # Neck→R_Shoulder, propagate
            ((8, 9), 9, [10]),           # R_Shoulder→R_Elbow
            ((9, 10), 10, []),           # R_Elbow→R_F_Paw

            # Left back leg
            ((4, 11), 11, [12, 13]),     # Tail→L_Hip
            ((11, 12), 12, [13]),        # L_Hip→L_Knee
            ((12, 13), 13, []),          # L_Knee→L_B_Paw

            # Right back leg
            ((4, 14), 14, [15, 16]),     # Tail→R_Hip
            ((14, 15), 15, [16]),        # R_Hip→R_Knee
            ((15, 16), 16, []),          # R_Knee→R_B_Paw

            # Head: eyes
            ((2, 0), 0, []),             # Nose→L_Eye
            ((2, 1), 1, []),             # Nose→R_Eye
        ]

        for (p1, p2), end_joint, propagate_joints in bone_chain:
            ratio = ratios.get((p1, p2), 1.0)
            x_offset = (candidate[p1][0] - candidate[p2][0]) * (1.0 - ratio)
            y_offset = (candidate[p1][1] - candidate[p2][1]) * (1.0 - ratio)

            # Move end joint and all downstream joints
            joints_to_move = [end_joint] + propagate_joints
            for j in joints_to_move:
                if j < len(candidate):
                    candidate[j, 0] += x_offset
                    candidate[j, 1] += y_offset

    return detected_poses
