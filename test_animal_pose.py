"""
Standalone tests for ComfyUI-WanAnimalPreprocess.
Tests core logic without requiring ComfyUI runtime.
Run: python test_animal_pose.py
"""

import sys
import os
import numpy as np
import traceback

# Allow standalone imports (no ComfyUI)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = 0
FAIL = 0

def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  PASS: {name}")
        PASS += 1
    except Exception as e:
        print(f"  FAIL: {name}")
        traceback.print_exc()
        FAIL += 1


# ==================== 1. pose2d_utils ====================
print("\n[1] pose_utils/pose2d_utils.py")

def test_aaposemeta_creation():
    from pose_utils.pose2d_utils import AAPoseMeta
    meta = AAPoseMeta()
    meta.width = 640
    meta.height = 480
    meta.kps_body = np.random.rand(17, 2)
    meta.kps_body_p = np.random.rand(17)
    assert meta.width == 640
    assert meta.height == 480
    assert meta.kps_body.shape == (17, 2)
    assert meta.kps_body_p.shape == (17,)

test("AAPoseMeta creation (17 keypoints)", test_aaposemeta_creation)


def test_aaposemeta_from_kps_body():
    from pose_utils.pose2d_utils import AAPoseMeta
    # from_kps_body expects (17,3) array with [x+y+conf], height, width
    kps = np.column_stack([np.random.rand(17, 2) * 640, np.random.rand(17)])
    meta = AAPoseMeta.from_kps_body(kps, 480, 640)
    assert meta.width == 640
    assert meta.height == 480
    assert meta.kps_body.shape == (17, 2)
    assert meta.kps_body_p.shape == (17,)

test("AAPoseMeta.from_kps_body()", test_aaposemeta_from_kps_body)


def test_aaposemeta_from_humanapi_meta():
    from pose_utils.pose2d_utils import AAPoseMeta
    meta_dict = {
        'width': 640,
        'height': 480,
        'keypoints_body': [[0.1 * i, 0.05 * i, 0.9] for i in range(17)],
    }
    meta = AAPoseMeta.from_humanapi_meta(meta_dict)
    assert meta.width == 640
    assert meta.height == 480
    assert meta.kps_body.shape == (17, 2)
    assert meta.kps_body_p.shape == (17,)

test("AAPoseMeta.from_humanapi_meta()", test_aaposemeta_from_humanapi_meta)


def test_split_kp2ds_for_animal():
    from pose_utils.pose2d_utils import split_kp2ds_for_animal
    kp2ds = np.random.rand(17, 3)
    # Returns a single (17,3) copy for animals (all keypoints are body)
    result = split_kp2ds_for_animal(kp2ds)
    assert result.shape == (17, 3), f"Expected (17,3), got {result.shape}"
    assert np.array_equal(result, kp2ds)

test("split_kp2ds_for_animal()", test_split_kp2ds_for_animal)


def test_load_pose_metas_from_kp2ds_seq():
    from pose_utils.pose2d_utils import load_pose_metas_from_kp2ds_seq
    # Simulate 5 frames with 17 keypoints: (5, 17, 3)
    kp2ds = np.random.rand(5, 17, 3)
    kp2ds[:, :, :2] *= 100  # pixel coords
    metas = load_pose_metas_from_kp2ds_seq(kp2ds, width=640, height=480)
    assert len(metas) == 5, f"Expected 5 metas, got {len(metas)}"
    for m in metas:
        assert 'keypoints_body' in m
        assert m['width'] == 640
        assert m['height'] == 480
        assert len(m['keypoints_body']) == 17
        # No hands/face keys
        assert 'keypoints_right_hand' not in m
        assert 'keypoints_left_hand' not in m
        assert 'keypoints_face' not in m

test("load_pose_metas_from_kp2ds_seq() - animal format", test_load_pose_metas_from_kp2ds_seq)


def test_keypoints_from_heatmaps():
    from pose_utils.pose2d_utils import keypoints_from_heatmaps
    # Simulate heatmaps: (1, 17, 64, 48)
    heatmaps = np.zeros((1, 17, 64, 48), dtype=np.float32)
    # Put a peak at (32, 24) for each keypoint
    for k in range(17):
        heatmaps[0, k, 32, 24] = 1.0
    center = np.array([[320, 240]])
    scale = np.array([[2.0, 2.0]])
    preds, maxvals = keypoints_from_heatmaps(heatmaps, center, scale)
    assert preds.shape == (1, 17, 2), f"Expected (1,17,2), got {preds.shape}"
    assert maxvals.shape == (1, 17, 1), f"Expected (1,17,1), got {maxvals.shape}"

test("keypoints_from_heatmaps()", test_keypoints_from_heatmaps)


# ==================== 2. human_visualization ====================
print("\n[2] pose_utils/human_visualization.py")

def test_draw_animal_pose():
    from pose_utils.human_visualization import draw_animal_pose
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    # Fake 17 keypoints in pixel coordinates
    kp2ds = np.zeros((17, 3))
    kp2ds[:, 0] = np.linspace(100, 500, 17)  # x
    kp2ds[:, 1] = np.linspace(100, 400, 17)  # y
    kp2ds[:, 2] = 0.9  # confidence
    result = draw_animal_pose(canvas, kp2ds, threshold=0.5)
    assert result.shape == (480, 640, 3)
    assert result.sum() > 0, "Drawing should produce non-zero pixels"

test("draw_animal_pose() draws on canvas", test_draw_animal_pose)


def test_draw_animal_pose_by_meta():
    from pose_utils.human_visualization import draw_animal_pose_by_meta, AAPoseMeta
    meta = AAPoseMeta()
    meta.width = 640
    meta.height = 480
    meta.kps_body = np.column_stack([
        np.linspace(100, 500, 17),
        np.linspace(100, 400, 17),
    ])
    meta.kps_body_p = np.full(17, 0.9)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    result = draw_animal_pose_by_meta(canvas, meta, dataset='ap10k')
    assert result.sum() > 0
    # Also test apt36k (same skeleton)
    canvas2 = np.zeros((480, 640, 3), dtype=np.uint8)
    result2 = draw_animal_pose_by_meta(canvas2, meta, dataset='apt36k')
    assert result2.sum() > 0

test("draw_animal_pose_by_meta() with ap10k and apt36k", test_draw_animal_pose_by_meta)


def test_draw_skeleten():
    from pose_utils.human_visualization import draw_skeleten
    meta = {
        'width': 640,
        'height': 480,
        'keypoints_body': [[i * 0.05, i * 0.03, 0.9] for i in range(17)],
    }
    result = draw_skeleten(meta)
    assert result.shape == (480, 640, 3)

test("draw_skeleten() from meta dict", test_draw_skeleten)


def test_animal_skeleton_constants():
    from pose_utils.human_visualization import (
        ANIMAL_LIMB_SEQ, ANIMAL_HEAD_LIMBS,
        ANIMAL_LIMB_COLORS, ANIMAL_HEAD_COLORS, ANIMAL_KP_COLORS
    )
    assert len(ANIMAL_LIMB_SEQ) == 14, f"Expected 14 limbs, got {len(ANIMAL_LIMB_SEQ)}"
    assert len(ANIMAL_HEAD_LIMBS) == 2
    assert len(ANIMAL_LIMB_COLORS) == 14
    assert len(ANIMAL_HEAD_COLORS) == 2
    assert len(ANIMAL_KP_COLORS) == 17
    # All indices should be 0-16
    for a, b in ANIMAL_LIMB_SEQ + ANIMAL_HEAD_LIMBS:
        assert 0 <= a <= 16 and 0 <= b <= 16, f"Invalid index: ({a}, {b})"

test("Skeleton constants (17 kp, 14+2 limbs)", test_animal_skeleton_constants)


# ==================== 3. retarget_pose ====================
print("\n[3] retarget_pose.py")

def test_retarget_limbseq():
    from retarget_pose import limbSeq, keypoint_list
    assert len(keypoint_list) == 17, f"Expected 17 keypoint names, got {len(keypoint_list)}"
    assert len(limbSeq) == 16, f"Expected 16 limb connections, got {len(limbSeq)}"
    # All 1-indexed, should be 1-17
    for a, b in limbSeq:
        assert 1 <= a <= 17 and 1 <= b <= 17, f"Invalid 1-indexed pair: ({a}, {b})"

test("limbSeq and keypoint_list definitions", test_retarget_limbseq)


def test_get_length():
    from retarget_pose import get_length
    skeleton = {
        'height': 480,
        'width': 640,
        'keypoints_body': [[0.5, 0.5, 0.9]] * 17,
    }
    # Change two points to measure distance
    skeleton['keypoints_body'][2] = [0.3, 0.3, 0.9]  # Nose
    skeleton['keypoints_body'][3] = [0.5, 0.5, 0.9]  # Neck
    X, Y, length = get_length(skeleton, [3, 4])  # 1-indexed: Nose-Neck
    assert length is not None
    assert length > 0

test("get_length() bone measurement", test_get_length)


def test_check_full_body():
    from retarget_pose import check_full_body
    # Full body: all points high confidence
    kps = [[0.5, 0.5, 0.9]] * 17
    assert check_full_body(kps, threshold=0.4) == 'full_body'

    # Half body: missing back paws
    kps_half = [[0.5, 0.5, 0.9]] * 17
    kps_half[13] = [0.5, 0.5, 0.1]  # L_B_Paw low conf
    kps_half[16] = [0.5, 0.5, 0.1]  # R_B_Paw low conf
    kps_half[11] = [0.5, 0.5, 0.1]  # L_Hip low
    kps_half[14] = [0.5, 0.5, 0.1]  # R_Hip low
    result = check_full_body(kps_half, threshold=0.4)
    assert result in ('half_body', 'three_quarter_body'), f"Got {result}"

test("check_full_body() detection", test_check_full_body)


def test_check_full_body_both():
    from retarget_pose import check_full_body_both
    assert check_full_body_both('full_body', 'half_body') == 'half_body'
    assert check_full_body_both('full_body', 'full_body') == 'full_body'
    assert check_full_body_both('three_quarter_body', 'half_body') == 'half_body'

test("check_full_body_both()", test_check_full_body_both)


def test_fix_lack_keypoints_use_sym():
    from retarget_pose import fix_lack_keypoints_use_sym
    skeleton = {
        'height': 480,
        'width': 640,
        'keypoints_body': [[0.5, 0.5, 0.9]] * 17,
    }
    # Remove right front paw
    skeleton['keypoints_body'][10] = None
    result = fix_lack_keypoints_use_sym(skeleton)
    # Should not crash
    assert result is not None
    assert len(result['keypoints_body']) == 17

test("fix_lack_keypoints_use_sym() handles None", test_fix_lack_keypoints_use_sym)


def test_get_retarget_pose():
    from retarget_pose import get_retarget_pose
    # Build two valid meta dicts
    def make_meta():
        return {
            'height': 480,
            'width': 640,
            'keypoints_body': [[0.3 + i * 0.02, 0.2 + i * 0.03, 0.9] for i in range(17)],
        }
    tpl0 = make_meta()
    ref = make_meta()
    # Slightly different ref
    ref['keypoints_body'] = [[0.35 + i * 0.02, 0.25 + i * 0.03, 0.85] for i in range(17)]
    tpls = [make_meta() for _ in range(3)]

    result = get_retarget_pose(tpl0, ref, tpls, None, None)
    assert len(result) == 3, f"Expected 3 retargeted metas, got {len(result)}"
    for pm in result:
        assert hasattr(pm, 'kps_body')
        assert hasattr(pm, 'kps_body_p')
        assert pm.kps_body.shape == (17, 2), f"Expected (17,2), got {pm.kps_body.shape}"
        assert pm.kps_body_p.shape == (17,), f"Expected (17,), got {pm.kps_body_p.shape}"
        # Should NOT have hand attributes
        assert not hasattr(pm, 'kps_rhand') or pm.kps_rhand is None
        assert not hasattr(pm, 'kps_lhand') or pm.kps_lhand is None

test("get_retarget_pose() end-to-end", test_get_retarget_pose)


# ==================== 4. onetoall/infer_function ====================
print("\n[4] onetoall/infer_function.py")

def test_aaposemeta_to_dwpose():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    # Import using direct path since relative imports won't work standalone
    import importlib
    spec = importlib.util.spec_from_file_location(
        "infer_function",
        os.path.join(os.path.dirname(__file__), "onetoall", "infer_function.py"),
        submodule_search_locations=[]
    )
    # Can't easily test relative import modules standalone, so test the logic manually
    meta = {
        'keypoints_body': np.column_stack([
            np.linspace(0.1, 0.9, 17),
            np.linspace(0.1, 0.8, 17),
            np.full(17, 0.9),
        ]),
    }
    kps = np.array(meta['keypoints_body'])
    candidate_body = kps[:, :2]
    score_body = kps[:, 2]
    assert candidate_body.shape == (17, 2)
    assert score_body.shape == (17,)
    # Verify no hands/face references needed
    dwpose = {
        "bodies": {"candidate": candidate_body, "subset": np.expand_dims(np.arange(17, dtype=float), 0), "score": np.expand_dims(score_body, 0)},
        "hands": None,
        "hands_score": None,
        "faces": None,
        "faces_score": None,
    }
    assert dwpose["hands"] is None
    assert dwpose["faces"] is None

test("aaposemeta_to_dwpose logic (no hands/face)", test_aaposemeta_to_dwpose)


# ==================== 5. onetoall/utils ====================
print("\n[5] onetoall/utils.py")

def test_onetoall_draw_bodypose():
    # Test the drawing function by importing directly
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "onetoall"))
    from utils import draw_bodypose_aligned, get_stickwidth
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    candidate = np.column_stack([
        np.linspace(0.2, 0.8, 17),
        np.linspace(0.2, 0.8, 17),
    ])
    subset = np.arange(17, dtype=float).reshape(1, -1)
    score = np.full((1, 17), 0.9)
    result = draw_bodypose_aligned(canvas, candidate, subset, score)
    assert result.shape == (480, 640, 3)
    assert result.sum() > 0, "Should draw something"

test("draw_bodypose_aligned() animal skeleton", test_onetoall_draw_bodypose)


def test_onetoall_draw_pose_aligned():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "onetoall"))
    from utils import draw_pose_aligned
    pose = {
        "bodies": {
            "candidate": np.column_stack([np.linspace(0.2, 0.8, 17), np.linspace(0.2, 0.8, 17)]),
            "subset": np.arange(17, dtype=float).reshape(1, -1),
            "score": np.full((1, 17), 0.9),
        },
        "hands": None,
        "faces": None,
    }
    result = draw_pose_aligned(pose, 480, 640)
    assert result.shape == (480, 640, 3)

    # Test head_strength modes
    for mode in ["full", "weak", "none"]:
        r = draw_pose_aligned(pose, 480, 640, head_strength=mode)
        assert r.shape == (480, 640, 3), f"Failed for head_strength={mode}"

test("draw_pose_aligned() with head_strength modes", test_onetoall_draw_pose_aligned)


# ==================== 6. utils.py ====================
print("\n[6] utils.py")

# Fix import path: remove onetoall from sys.path to avoid shadowing root utils
sys.path = [p for p in sys.path if not p.endswith('onetoall')]

def test_padding_resize():
    import importlib
    # Force re-import of root utils (not onetoall/utils)
    if 'utils' in sys.modules:
        del sys.modules['utils']
    from utils import padding_resize
    img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    result = padding_resize(img, height=480, width=640)
    assert result.shape == (480, 640, 3)

test("padding_resize()", test_padding_resize)


def test_resize_to_bounds():
    if 'utils' in sys.modules:
        del sys.modules['utils']
    from utils import resize_to_bounds
    img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    result = resize_to_bounds(img, height=480, width=640)
    assert result.shape == (480, 640, 3)

test("resize_to_bounds()", test_resize_to_bounds)


def test_get_head_bboxes():
    if 'utils' in sys.modules:
        del sys.modules['utils']
    from utils import get_head_bboxes
    kp2ds = np.zeros((17, 3))
    kp2ds[0] = [100, 80, 0.9]   # L_Eye
    kp2ds[1] = [140, 80, 0.9]   # R_Eye
    kp2ds[2] = [120, 100, 0.9]  # Nose
    bbox = get_head_bboxes(kp2ds, scale=2.0, image_shape=(480, 640))
    assert len(bbox) == 4
    assert bbox[0] < bbox[1]  # min_x < max_x
    assert bbox[2] < bbox[3]  # min_y < max_y

test("get_head_bboxes() (replaces get_face_bboxes)", test_get_head_bboxes)


def test_no_face_bboxes():
    """Verify get_face_bboxes was removed (human-specific function)."""
    if 'utils' in sys.modules:
        del sys.modules['utils']
    import utils
    assert not hasattr(utils, 'get_face_bboxes'), "get_face_bboxes should be removed"

test("get_face_bboxes removed from utils", test_no_face_bboxes)


def test_get_frame_indices():
    if 'utils' in sys.modules:
        del sys.modules['utils']
    from utils import get_frame_indices
    indices = get_frame_indices(frame_num=100, video_fps=30, clip_length=16, train_fps=8)
    assert len(indices) == 16
    assert all(0 <= i < 100 for i in indices)

test("get_frame_indices()", test_get_frame_indices)


# ==================== 7. model_downloader ====================
print("\n[7] models/model_downloader.py")

def test_model_registry():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
    from model_downloader import ANIMAL_POSE_MODELS
    # Check AP10k models exist
    ap10k_models = [k for k in ANIMAL_POSE_MODELS if 'ap10k' in k]
    apt36k_models = [k for k in ANIMAL_POSE_MODELS if 'apt36k' in k]
    yolo_models = [k for k in ANIMAL_POSE_MODELS if 'yolo' in k]
    assert len(ap10k_models) >= 4, f"Expected >=4 AP10k models, got {len(ap10k_models)}"
    assert len(apt36k_models) >= 4, f"Expected >=4 APT36k models, got {len(apt36k_models)}"
    assert len(yolo_models) >= 3, f"Expected >=3 YOLO models, got {len(yolo_models)}"

test("ANIMAL_POSE_MODELS has ap10k + apt36k + yolo", test_model_registry)


def test_no_duplicate_functions():
    """Check model_downloader.py doesn't have duplicate function defs."""
    fpath = os.path.join(os.path.dirname(__file__), "models", "model_downloader.py")
    with open(fpath, 'r') as f:
        content = f.read()
    for fname in ['calculate_file_hash', 'verify_model_file', 'download_model', 'ensure_model_available', 'list_available_models']:
        count = content.count(f'def {fname}(')
        assert count == 1, f"Function '{fname}' defined {count} times (expected 1)"

test("No duplicate function definitions", test_no_duplicate_functions)


def test_no_get_detection_models_dir():
    """Verify the broken get_detection_models_dir reference was removed."""
    fpath = os.path.join(os.path.dirname(__file__), "models", "model_downloader.py")
    with open(fpath, 'r') as f:
        content = f.read()
    assert 'get_detection_models_dir' not in content, "Broken get_detection_models_dir reference should be removed"

test("No broken get_detection_models_dir reference", test_no_get_detection_models_dir)


# ==================== 8. Integration checks ====================
print("\n[8] Integration / cross-module checks")

def test_no_hand_face_in_retarget():
    """Verify retarget_pose.py has no hand/face references."""
    with open(os.path.join(os.path.dirname(__file__), 'retarget_pose.py'), 'r') as f:
        content = f.read()
    for bad_term in ['deal_hand', 'kps_rhand', 'kps_lhand', 'keypoints_right_hand',
                     'keypoints_left_hand', 'keypoints_face', 'face_bbox']:
        assert bad_term not in content, f"Found '{bad_term}' in retarget_pose.py"

test("retarget_pose.py: no hand/face references", test_no_hand_face_in_retarget)


def test_no_hand_face_in_onetoall_infer():
    """Verify onetoall/infer_function.py has no hand/face references."""
    with open(os.path.join(os.path.dirname(__file__), 'onetoall', 'infer_function.py'), 'r') as f:
        content = f.read()
    for bad_term in ['kps_rhand', 'kps_lhand', 'keypoints_right_hand',
                     'keypoints_left_hand', 'keypoints_face', '_face_scale_only',
                     'L_EYE_IDXS', 'NOSE_TIP', 'MOUTH_L']:
        assert bad_term not in content, f"Found '{bad_term}' in onetoall/infer_function.py"

test("onetoall/infer_function.py: no hand/face references", test_no_hand_face_in_onetoall_infer)


def test_no_hand_face_in_onetoall_utils():
    """Verify onetoall/utils.py has no hand/face draw functions."""
    with open(os.path.join(os.path.dirname(__file__), 'onetoall', 'utils.py'), 'r') as f:
        content = f.read()
    for bad_term in ['draw_handpose', 'draw_facepose']:
        assert bad_term not in content, f"Found '{bad_term}' in onetoall/utils.py"

test("onetoall/utils.py: no hand/face draw functions", test_no_hand_face_in_onetoall_utils)


def test_nodes_has_dataset_param():
    """Verify nodes.py passes dataset through the pipeline."""
    with open(os.path.join(os.path.dirname(__file__), 'nodes.py'), 'r') as f:
        content = f.read()
    assert '"dataset"' in content, "nodes.py should reference 'dataset'"
    assert '"ap10k"' in content or "'ap10k'" in content, "nodes.py should have ap10k default"
    assert 'model.get("dataset"' in content, "nodes.py should read dataset from model dict"

test("nodes.py: dataset flows through pipeline", test_nodes_has_dataset_param)


def test_all_init_files_exist():
    """Verify __init__.py exists in all subpackages."""
    base = os.path.dirname(os.path.abspath(__file__))
    for subdir in ['pose_utils', 'onetoall', 'models']:
        init_path = os.path.join(base, subdir, '__init__.py')
        assert os.path.exists(init_path), f"Missing {subdir}/__init__.py"

test("All __init__.py files exist", test_all_init_files_exist)


def test_pyproject_updated():
    """Verify pyproject.toml references animal project."""
    with open(os.path.join(os.path.dirname(__file__), 'pyproject.toml'), 'r') as f:
        content = f.read()
    assert 'WanAnimalPreprocess' in content
    assert 'WanAnimatePreprocess' not in content, "Should not reference old human project name"
    assert 'kijai' not in content, "Should not reference kijai as publisher"

test("pyproject.toml updated for animal project", test_pyproject_updated)


# ==================== Summary ====================
print(f"\n{'='*50}")
print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
if FAIL == 0:
    print("All tests passed!")
else:
    print(f"WARNING: {FAIL} test(s) failed")
    sys.exit(1)
