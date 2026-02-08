# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified for animal pose retargeting (AP10k/APT36k format - 17 keypoints, no hands/face).

import numpy as np
from tqdm import tqdm
import math
from typing import NamedTuple
import copy

try:
    from .pose_utils.pose2d_utils import AAPoseMeta
except ImportError:
    from pose_utils.pose2d_utils import AAPoseMeta

# ==================== Animal Pose Keypoint Definitions ====================
# AP10k / APT36k format (17 keypoints):
# 0: L_Eye, 1: R_Eye, 2: Nose, 3: Neck, 4: Root_of_tail
# 5: L_Shoulder, 6: L_Elbow, 7: L_F_Paw
# 8: R_Shoulder, 9: R_Elbow, 10: R_F_Paw
# 11: L_Hip, 12: L_Knee, 13: L_B_Paw
# 14: R_Hip, 15: R_Knee, 16: R_B_Paw

keypoint_list = [
    "L_Eye", "R_Eye", "Nose", "Neck", "Root_of_tail",
    "L_Shoulder", "L_Elbow", "L_F_Paw",
    "R_Shoulder", "R_Elbow", "R_F_Paw",
    "L_Hip", "L_Knee", "L_B_Paw",
    "R_Hip", "R_Knee", "R_B_Paw",
]

# Skeleton connections (1-indexed) for bone length calculation and retargeting
# Using 1-indexed to match the original kijai convention
limbSeq = [
    [3, 4],              # Nose to Neck
    [4, 5],              # Neck to Tail (spine)
    [4, 6], [6, 7], [7, 8],      # Left front leg (Neck→L_Shoulder→L_Elbow→L_F_Paw)
    [4, 9], [9, 10], [10, 11],   # Right front leg (Neck→R_Shoulder→R_Elbow→R_F_Paw)
    [5, 12], [12, 13], [13, 14], # Left back leg (Tail→L_Hip→L_Knee→L_B_Paw)
    [5, 15], [15, 16], [16, 17], # Right back leg (Tail→R_Hip→R_Knee→R_B_Paw)
    [3, 1], [3, 2],              # Nose to Eyes
]

eps = 0.01


class Keypoint(NamedTuple):
    x: float
    y: float
    score: float = 1.0
    id: int = -1


# ==================== Bone Length Utilities ====================

def get_length(skeleton, limb):
    """Calculate bone length for a given limb in a skeleton."""
    k1_index, k2_index = limb
    H, W = skeleton['height'], skeleton['width']
    keypoints = skeleton['keypoints_body']
    keypoint1 = keypoints[k1_index - 1]
    keypoint2 = keypoints[k2_index - 1]

    if keypoint1 is None or keypoint2 is None:
        return None, None, None

    X = np.array([keypoint1[0], keypoint2[0]]) * float(W)
    Y = np.array([keypoint1[1], keypoint2[1]]) * float(H)
    length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5

    return X, Y, length


# ==================== Pose Scaling ====================

def get_scaled_pose(canvas, src_canvas, keypoints, bone_ratio_list,
                    delta_ground_x, delta_ground_y, rescaled_src_ground_x,
                    body_flag, id, scale_min, threshold=0.4):
    """Scale and translate animal pose keypoints."""
    H, W = canvas
    src_H, src_W = src_canvas

    new_length_list = []
    angle_list = []

    # Keypoints from 0-1 to H/W range
    for idx in range(len(keypoints)):
        if keypoints[idx] is None or len(keypoints[idx]) == 0:
            continue
        keypoints[idx] = [keypoints[idx][0] * src_W, keypoints[idx][1] * src_H, keypoints[idx][2]]

    # First pass: get new_length_list and angle_list
    for idx, (k1_index, k2_index) in enumerate(limbSeq):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None or len(keypoint1) == 0 or len(keypoint2) == 0:
            new_length_list.append(None)
            angle_list.append(None)
            continue

        Y = np.array([keypoint1[0], keypoint2[0]])
        X = np.array([keypoint1[1], keypoint2[1]])
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        new_length = length * bone_ratio_list[idx]
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))

        new_length_list.append(new_length)
        angle_list.append(angle)

    # Second pass: calculate new keypoints
    rescale_keypoints = keypoints.copy()

    for idx, (k1_index, k2_index) in enumerate(limbSeq):
        start_keypoint = rescale_keypoints[k1_index - 1]
        new_length = new_length_list[idx]
        angle = angle_list[idx]

        if rescale_keypoints[k1_index - 1] is None or rescale_keypoints[k2_index - 1] is None or \
           len(rescale_keypoints[k1_index - 1]) == 0 or len(rescale_keypoints[k2_index - 1]) == 0:
            continue

        delta_x = new_length * math.cos(math.radians(angle))
        delta_y = new_length * math.sin(math.radians(angle))

        end_keypoint_x = start_keypoint[0] - delta_x
        end_keypoint_y = start_keypoint[1] - delta_y

        rescale_keypoints[k2_index - 1] = [end_keypoint_x, end_keypoint_y, rescale_keypoints[k2_index - 1][2]]

    # Global position offset for first frame
    if id == 0:
        # Use hips (L_Hip=11, R_Hip=14) as ground reference for full body
        if body_flag == 'full_body' and rescale_keypoints[11] is not None and rescale_keypoints[14] is not None:
            delta_ground_x_offset = (rescale_keypoints[11][0] + rescale_keypoints[14][0]) / 2 - rescaled_src_ground_x
            delta_ground_x += delta_ground_x_offset
        elif body_flag == 'half_body' and rescale_keypoints[3] is not None:
            # Use Neck as anchor for half body
            delta_ground_x_offset = rescale_keypoints[3][0] - rescaled_src_ground_x
            delta_ground_x += delta_ground_x_offset

    # Offset all keypoints
    for idx in range(len(rescale_keypoints)):
        if rescale_keypoints[idx] is None or len(rescale_keypoints[idx]) == 0:
            continue
        rescale_keypoints[idx][0] -= delta_ground_x
        rescale_keypoints[idx][1] -= delta_ground_y
        rescale_keypoints[idx][0] /= scale_min
        rescale_keypoints[idx][1] /= scale_min

    # Get normalized keypoints_body
    norm_body_keypoints = []
    for body_keypoint in rescale_keypoints:
        if body_keypoint is not None:
            norm_body_keypoints.append([body_keypoint[0] / W, body_keypoint[1] / H, body_keypoint[2]])
        else:
            norm_body_keypoints.append(None)

    frame_info = {
        'height': H,
        'width': W,
        'keypoints_body': norm_body_keypoints,
    }

    return frame_info


def rescale_skeleton(H, W, keypoints, bone_ratio_list):
    """Rescale skeleton according to bone ratio list."""
    rescale_keypoints = keypoints.copy()

    new_length_list = []
    angle_list = []

    for idx in range(len(rescale_keypoints)):
        if rescale_keypoints[idx] is None or len(rescale_keypoints[idx]) == 0:
            continue
        rescale_keypoints[idx] = [rescale_keypoints[idx][0] * W, rescale_keypoints[idx][1] * H]

    for idx, (k1_index, k2_index) in enumerate(limbSeq):
        keypoint1 = rescale_keypoints[k1_index - 1]
        keypoint2 = rescale_keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None or len(keypoint1) == 0 or len(keypoint2) == 0:
            new_length_list.append(None)
            angle_list.append(None)
            continue

        Y = np.array([keypoint1[0], keypoint2[0]])
        X = np.array([keypoint1[1], keypoint2[1]])
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        new_length = length * bone_ratio_list[idx]
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))

        new_length_list.append(new_length)
        angle_list.append(angle)

    for idx, (k1_index, k2_index) in enumerate(limbSeq):
        start_keypoint = rescale_keypoints[k1_index - 1]
        new_length = new_length_list[idx]
        angle = angle_list[idx]

        if rescale_keypoints[k1_index - 1] is None or rescale_keypoints[k2_index - 1] is None or \
           len(rescale_keypoints[k1_index - 1]) == 0 or len(rescale_keypoints[k2_index - 1]) == 0:
            continue

        delta_x = new_length * math.cos(math.radians(angle))
        delta_y = new_length * math.sin(math.radians(angle))

        end_keypoint_x = start_keypoint[0] - delta_x
        end_keypoint_y = start_keypoint[1] - delta_y

        rescale_keypoints[k2_index - 1] = [end_keypoint_x, end_keypoint_y]

    return rescale_keypoints


def fix_lack_keypoints_use_sym(skeleton):
    """Fix missing keypoints using symmetric limbs (animal version)."""
    keypoints = skeleton['keypoints_body']
    H, W = skeleton['height'], skeleton['width']

    # Symmetric limb pairs (left vs right)
    # Left front: [5,6,7], Right front: [8,9,10]
    # Left back: [11,12,13], Right back: [14,15,16]
    limb_points_list = [
        [5, 6, 7],     # Left front leg
        [8, 9, 10],    # Right front leg
        [11, 12, 13],  # Left back leg
        [14, 15, 16],  # Right back leg
    ]

    for limb_points in limb_points_list:
        miss_flag = False
        for point in limb_points:
            if keypoints[point] is None:
                miss_flag = True
                continue
            if miss_flag:
                skeleton['keypoints_body'][point] = None

    # Symmetric repair: left front <-> right front, left back <-> right back
    repair_limb_seq_left = [
        [5, 6], [6, 7],      # Left front leg
        [11, 12], [12, 13],  # Left back leg
    ]
    repair_limb_seq_right = [
        [8, 9], [9, 10],     # Right front leg
        [14, 15], [15, 16],  # Right back leg
    ]

    repair_limb_seq = [repair_limb_seq_left, repair_limb_seq_right]

    for idx_part, part in enumerate(repair_limb_seq):
        for idx, limb in enumerate(part):
            k1_index, k2_index = limb
            keypoint1 = keypoints[k1_index]
            keypoint2 = keypoints[k2_index]

            if keypoint1 is not None and keypoint2 is None:
                sym_limb = repair_limb_seq[1 - idx_part][idx]
                k1_index_sym, k2_index_sym = sym_limb
                keypoint1_sym = keypoints[k1_index_sym]
                keypoint2_sym = keypoints[k2_index_sym]
                ref_length = 0

                if keypoint1_sym is not None and keypoint2_sym is not None:
                    X = np.array([keypoint1_sym[0], keypoint2_sym[0]]) * float(W)
                    Y = np.array([keypoint1_sym[1], keypoint2_sym[1]]) * float(H)
                    ref_length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                else:
                    # Use spine (Neck to Tail) as reference
                    if keypoints[3] is not None and keypoints[4] is not None:
                        X = np.array([keypoints[3][0], keypoints[4][0]]) * float(W)
                        Y = np.array([keypoints[3][1], keypoints[4][1]]) * float(H)
                        ref_length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                        ref_length /= 3  # Approximate limb length as fraction of spine

                if ref_length != 0:
                    skeleton['keypoints_body'][k2_index] = [0, 0, 0.5]
                    skeleton['keypoints_body'][k2_index][0] = skeleton['keypoints_body'][k1_index][0]
                    skeleton['keypoints_body'][k2_index][1] = skeleton['keypoints_body'][k1_index][1] + ref_length / H

    return skeleton


def rescale_shorten_skeleton(ratio_list, src_length_list, dst_length_list):
    """Rescale to keep symmetric limbs similar in length."""
    # Symmetric bone pairs (indices into limbSeq)
    modify_bone_list = [
        [2, 5],   # Neck→L_Shoulder vs Neck→R_Shoulder
        [3, 6],   # L_Shoulder→L_Elbow vs R_Shoulder→R_Elbow
        [4, 7],   # L_Elbow→L_F_Paw vs R_Elbow→R_F_Paw
        [8, 11],  # Tail→L_Hip vs Tail→R_Hip
        [9, 12],  # L_Hip→L_Knee vs R_Hip→R_Knee
        [10, 13], # L_Knee→L_B_Paw vs R_Knee→R_B_Paw
    ]

    for modify_bone in modify_bone_list:
        idx_a, idx_b = modify_bone
        if idx_a < len(ratio_list) and idx_b < len(ratio_list):
            if ratio_list[idx_a] is not None and ratio_list[idx_b] is not None:
                new_ratio = max(ratio_list[idx_a], ratio_list[idx_b])
                ratio_list[idx_a] = new_ratio
                ratio_list[idx_b] = new_ratio

    # Symmetrize eye connections
    if len(ratio_list) > 15:
        if ratio_list[14] is not None and ratio_list[15] is not None:
            avg = (ratio_list[14] + ratio_list[15]) / 2
            ratio_list[14] = avg
            ratio_list[15] = avg

    return ratio_list, src_length_list, dst_length_list


def check_full_body(keypoints, threshold=0.4):
    """Check if the animal is fully visible (all four paws visible)."""
    body_flag = 'half_body'

    # Check if paw points exist with sufficient confidence
    # L_B_Paw(13), R_B_Paw(16) for back paws
    # L_F_Paw(7), R_F_Paw(10) for front paws
    front_paws_ok = (keypoints[7] is not None and keypoints[10] is not None and
                     keypoints[7][2] >= threshold and keypoints[10][2] >= threshold and
                     keypoints[7][1] <= 1 and keypoints[10][1] <= 1)
    back_paws_ok = (keypoints[13] is not None and keypoints[16] is not None and
                    keypoints[13][2] >= threshold and keypoints[16][2] >= threshold and
                    keypoints[13][1] <= 1 and keypoints[16][1] <= 1)

    if front_paws_ok and back_paws_ok:
        body_flag = 'full_body'
        return body_flag

    # Check if hips exist for three_quarter_body
    hips_ok = (keypoints[11] is not None and keypoints[14] is not None and
               keypoints[11][2] >= threshold and keypoints[14][2] >= threshold and
               keypoints[11][1] <= 1 and keypoints[14][1] <= 1)
    if hips_ok:
        body_flag = 'three_quarter_body'
        return body_flag

    return body_flag


def check_full_body_both(flag1, flag2):
    """Return the more restrictive body flag."""
    body_flag_dict = {'full_body': 2, 'three_quarter_body': 1, 'half_body': 0}
    body_flag_dict_reverse = {2: 'full_body', 1: 'three_quarter_body', 0: 'half_body'}

    flag1_num = body_flag_dict[flag1]
    flag2_num = body_flag_dict[flag2]
    return body_flag_dict_reverse[min(flag1_num, flag2_num)]


def write_to_poses(data_to_json, none_idx, dst_shape, bone_ratio_list,
                   delta_ground_x, delta_ground_y, rescaled_src_ground_x,
                   body_flag, scale_min):
    """Write retargeted poses for all frames."""
    outputs = []
    length = len(data_to_json)
    for id in tqdm(range(length)):
        src_height, src_width = data_to_json[id]['height'], data_to_json[id]['width']
        width, height = dst_shape
        keypoints = data_to_json[id]['keypoints_body']
        for idx in range(len(keypoints)):
            if idx in none_idx:
                keypoints[idx] = None
        new_keypoints = keypoints.copy()

        frame_info = get_scaled_pose(
            (height, width), (src_height, src_width),
            new_keypoints, bone_ratio_list,
            delta_ground_x, delta_ground_y,
            rescaled_src_ground_x, body_flag, id, scale_min,
        )
        outputs.append(frame_info)

    return outputs


def calculate_scale_ratio(skeleton, skeleton_edit, scale_ratio_flag):
    """Calculate scale ratio between skeleton and edited skeleton."""
    if scale_ratio_flag:
        # Use shoulder width ratio
        _, _, shoulder = get_length(skeleton, [4, 6])  # Neck to L_Shoulder
        _, _, shoulder_edit = get_length(skeleton_edit, [4, 6])
        if shoulder is not None and shoulder_edit is not None and shoulder_edit > 0:
            return shoulder / shoulder_edit
        return 1
    else:
        return 1


# ==================== Main Retarget Function ====================

def retarget_pose(src_skeleton, dst_skeleton, all_src_skeleton,
                  src_skeleton_edit, dst_skeleton_edit, threshold=0.4):
    """Retarget animal pose from source to destination skeleton."""
    if src_skeleton_edit is not None and dst_skeleton_edit is not None:
        use_edit_for_base = True
    else:
        use_edit_for_base = False

    src_skeleton_ori = copy.deepcopy(src_skeleton)
    dst_skeleton_ori_h, dst_skeleton_ori_w = dst_skeleton['height'], dst_skeleton['width']

    # Calculate global scale using Nose(2) and back paws (13, 16)
    def _calc_scale(src, dst):
        if (src['keypoints_body'][2] is not None and src['keypoints_body'][13] is not None and src['keypoints_body'][16] is not None and
            dst['keypoints_body'][2] is not None and dst['keypoints_body'][13] is not None and dst['keypoints_body'][16] is not None and
            src['keypoints_body'][2][2] > 0.5 and src['keypoints_body'][13][2] > 0.5 and
            dst['keypoints_body'][2][2] > 0.5 and dst['keypoints_body'][13][2] > 0.5):
            src_h = src['height'] * abs((src['keypoints_body'][13][1] + src['keypoints_body'][16][1]) / 2 - src['keypoints_body'][2][1])
            dst_h = dst['height'] * abs((dst['keypoints_body'][13][1] + dst['keypoints_body'][16][1]) / 2 - dst['keypoints_body'][2][1])
            return 1.0 * src_h / dst_h if dst_h > 0 else 1.0
        elif (src['keypoints_body'][3] is not None and src['keypoints_body'][4] is not None and
              dst['keypoints_body'][3] is not None and dst['keypoints_body'][4] is not None and
              src['keypoints_body'][3][2] > 0.5 and src['keypoints_body'][4][2] > 0.5 and
              dst['keypoints_body'][3][2] > 0.5 and dst['keypoints_body'][4][2] > 0.5):
            src_h = src['height'] * abs(src['keypoints_body'][4][1] - src['keypoints_body'][3][1])
            dst_h = dst['height'] * abs(dst['keypoints_body'][4][1] - dst['keypoints_body'][3][1])
            return 1.0 * src_h / dst_h if dst_h > 0 else 1.0
        else:
            return np.sqrt(src['height'] * src['width']) / np.sqrt(dst['height'] * dst['width'])

    scale_min = _calc_scale(src_skeleton, dst_skeleton)

    if use_edit_for_base:
        scale_ratio_flag = False
        scale_min_edit = _calc_scale(src_skeleton_edit, dst_skeleton_edit)
        ratio_src = calculate_scale_ratio(src_skeleton, src_skeleton_edit, scale_ratio_flag)
        ratio_dst = calculate_scale_ratio(dst_skeleton, dst_skeleton_edit, scale_ratio_flag)

        dst_skeleton_edit['height'] = int(dst_skeleton_edit['height'] * scale_min_edit)
        dst_skeleton_edit['width'] = int(dst_skeleton_edit['width'] * scale_min_edit)

    dst_skeleton['height'] = int(dst_skeleton['height'] * scale_min)
    dst_skeleton['width'] = int(dst_skeleton['width'] * scale_min)

    dst_body_flag = check_full_body(dst_skeleton['keypoints_body'], threshold)
    src_body_flag = check_full_body(src_skeleton_ori['keypoints_body'], threshold)
    body_flag = check_full_body_both(dst_body_flag, src_body_flag)

    if use_edit_for_base:
        src_skeleton_edit = fix_lack_keypoints_use_sym(src_skeleton_edit)
        dst_skeleton_edit = fix_lack_keypoints_use_sym(dst_skeleton_edit)
    else:
        src_skeleton = fix_lack_keypoints_use_sym(src_skeleton)
        dst_skeleton = fix_lack_keypoints_use_sym(dst_skeleton)

    none_idx = []
    for idx in range(len(dst_skeleton['keypoints_body'])):
        if dst_skeleton['keypoints_body'][idx] is None or src_skeleton['keypoints_body'][idx] is None:
            src_skeleton['keypoints_body'][idx] = None
            dst_skeleton['keypoints_body'][idx] = None
            none_idx.append(idx)

    # Get bone ratio list
    ratio_list, src_length_list, dst_length_list = [], [], []
    for idx, limb in enumerate(limbSeq):
        if use_edit_for_base:
            src_X, src_Y, src_length = get_length(src_skeleton_edit, limb)
            dst_X, dst_Y, dst_length = get_length(dst_skeleton_edit, limb)
            if src_X is None or dst_X is None:
                ratio = -1
            else:
                ratio = 1.0 * dst_length * ratio_dst / src_length / ratio_src
        else:
            src_X, src_Y, src_length = get_length(src_skeleton, limb)
            dst_X, dst_Y, dst_length = get_length(dst_skeleton, limb)
            if src_X is None or dst_X is None:
                ratio = -1
            else:
                ratio = 1.0 * dst_length / src_length

        ratio_list.append(ratio)
        src_length_list.append(src_length)
        dst_length_list.append(dst_length)

    for idx, ratio in enumerate(ratio_list):
        if ratio == -1:
            if ratio_list[0] != -1 and ratio_list[1] != -1:
                ratio_list[idx] = (ratio_list[0] + ratio_list[1]) / 2

    ratio_list, src_length_list, dst_length_list = rescale_shorten_skeleton(
        ratio_list, src_length_list, dst_length_list
    )

    rescaled_src_skeleton_ori = rescale_skeleton(
        src_skeleton_ori['height'], src_skeleton_ori['width'],
        src_skeleton_ori['keypoints_body'], ratio_list
    )

    # Get global translation offset
    if body_flag == 'full_body':
        # Use back paws as ground reference
        dst_ground_y = max(
            dst_skeleton['keypoints_body'][13][1],
            dst_skeleton['keypoints_body'][16][1]
        ) * dst_skeleton['height']
        rescaled_src_ground_y = max(rescaled_src_skeleton_ori[13][1], rescaled_src_skeleton_ori[16][1])
        delta_ground_y = rescaled_src_ground_y - dst_ground_y

        dst_ground_x = (dst_skeleton['keypoints_body'][11][0] + dst_skeleton['keypoints_body'][14][0]) * dst_skeleton['width'] / 2
        rescaled_src_ground_x = (rescaled_src_skeleton_ori[11][0] + rescaled_src_skeleton_ori[14][0]) / 2
        delta_ground_x = rescaled_src_ground_x - dst_ground_x
    else:
        # Use Neck as anchor
        src_neck_y = rescaled_src_skeleton_ori[3][1]
        dst_neck_y = dst_skeleton['keypoints_body'][3][1]
        delta_ground_y = src_neck_y - dst_neck_y * dst_skeleton['height']

        src_neck_x = rescaled_src_skeleton_ori[3][0]
        dst_neck_x = dst_skeleton['keypoints_body'][3][0]
        delta_ground_x = src_neck_x - dst_neck_x * dst_skeleton['width']
        rescaled_src_ground_x = src_neck_x

    dst_shape = (dst_skeleton_ori_w, dst_skeleton_ori_h)
    output = write_to_poses(
        all_src_skeleton, none_idx, dst_shape, ratio_list,
        delta_ground_x, delta_ground_y, rescaled_src_ground_x,
        body_flag, scale_min
    )
    return output


def get_retarget_pose(tpl_pose_meta0, refer_pose_meta, tpl_pose_metas,
                      tql_edit_pose_meta0, refer_edit_pose_meta):
    """Get retargeted pose metas for animal pose.

    Args:
        tpl_pose_meta0: First template pose meta dict
        refer_pose_meta: Reference pose meta dict
        tpl_pose_metas: All template pose meta dicts
        tql_edit_pose_meta0: Optional edited template (for Flux correction)
        refer_edit_pose_meta: Optional edited reference (for Flux correction)

    Returns:
        list[AAPoseMeta]: Retargeted pose meta objects
    """
    def _convert_meta(meta):
        for key, value in meta.items():
            if type(value) is np.ndarray:
                if not isinstance(value, list):
                    value = value.tolist()
            meta[key] = value
        return meta

    tpl_pose_meta0 = _convert_meta(copy.deepcopy(tpl_pose_meta0))
    refer_pose_meta = _convert_meta(copy.deepcopy(refer_pose_meta))

    tpl_pose_metas_new = []
    for meta in tpl_pose_metas:
        meta = _convert_meta(copy.deepcopy(meta))
        tpl_pose_metas_new.append(meta)

    if tql_edit_pose_meta0 is not None:
        tql_edit_pose_meta0 = _convert_meta(copy.deepcopy(tql_edit_pose_meta0))
    if refer_edit_pose_meta is not None:
        refer_edit_pose_meta = _convert_meta(copy.deepcopy(refer_edit_pose_meta))

    retarget_tpl_pose_metas = retarget_pose(
        tpl_pose_meta0, refer_pose_meta, tpl_pose_metas_new,
        tql_edit_pose_meta0, refer_edit_pose_meta
    )

    pose_metas = []
    for meta in retarget_tpl_pose_metas:
        pose_meta = AAPoseMeta()
        width, height = meta["width"], meta["height"]
        pose_meta.width = width
        pose_meta.height = height

        kps_body = []
        kps_body_p = []
        for kp in meta["keypoints_body"]:
            if kp is not None:
                kps_body.append([kp[0] * width, kp[1] * height])
                kps_body_p.append(kp[2] if len(kp) > 2 else 1.0)
            else:
                kps_body.append([0, 0])
                kps_body_p.append(0.0)

        pose_meta.kps_body = np.array(kps_body)
        pose_meta.kps_body_p = np.array(kps_body_p)

        pose_metas.append(pose_meta)

    return pose_metas

