# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Model downloader for Animal Pose Estimation models.
Automatically downloads AP10k/APT36k compatible models into ComfyUI models folder structure.
"""

import os
import urllib.request
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Model registry: model_name -> (url, expected_hash)
# Models are available from JunkyByte/easy_ViTPose on HuggingFace
# Official repo: https://huggingface.co/JunkyByte/easy_ViTPose
# Source: https://github.com/JunkyByte/easy_ViTPose

ANIMAL_POSE_MODELS = {
    # ===== AP10k Animal Pose Models =====
    # AP10k format: 17 keypoints for animals (dogs, cats, horses, livestock, etc.)
    # See: https://github.com/JunkyByte/easy_ViTPose#skeleton-reference
    
    "vitpose_s_ap10k.pth": {
        "url": "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/vitpose_s_ap10k.pth",
        "hash": None,  # Set SHA256 if available for verification
        "size_mb": 45,
        "description": "ViTPose Small for AP10k animal pose (17 keypoints, fastest)"
    },
    "vitpose_b_ap10k.pth": {
        "url": "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/vitpose_b_ap10k.pth",
        "hash": None,
        "size_mb": 90,
        "description": "ViTPose Base for AP10k animal pose (17 keypoints, balanced)"
    },
    "vitpose_l_ap10k.pth": {
        "url": "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/vitpose_l_ap10k.pth",
        "hash": None,
        "size_mb": 150,
        "description": "ViTPose Large for AP10k animal pose (17 keypoints, high quality)"
    },
    "vitpose_h_ap10k.pth": {
        "url": "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/vitpose_h_ap10k.pth",
        "hash": None,
        "size_mb": 300,
        "description": "ViTPose Huge for AP10k animal pose (17 keypoints, best quality)"
    },
    
    # ===== APT36k Animal Pose Models =====
    # APT36k format: 17 keypoints, alternative animal pose dataset
    # Same architecture as AP10k but trained on different dataset
    
    "vitpose_s_apt36k.pth": {
        "url": "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/vitpose_s_apt36k.pth",
        "hash": None,
        "size_mb": 45,
        "description": "ViTPose Small for APT36k animal pose (17 keypoints, fastest)"
    },
    "vitpose_b_apt36k.pth": {
        "url": "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/vitpose_b_apt36k.pth",
        "hash": None,
        "size_mb": 90,
        "description": "ViTPose Base for APT36k animal pose (17 keypoints, balanced)"
    },
    "vitpose_l_apt36k.pth": {
        "url": "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/vitpose_l_apt36k.pth",
        "hash": None,
        "size_mb": 150,
        "description": "ViTPose Large for APT36k animal pose (17 keypoints, high quality)"
    },
    "vitpose_h_apt36k.pth": {
        "url": "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/vitpose_h_apt36k.pth",
        "hash": None,
        "size_mb": 300,
        "description": "ViTPose Huge for APT36k animal pose (17 keypoints, best quality)"
    },
    
    # ===== YOLO Animal Detection Models =====
    # YOLOv8 models from Ultralytics
    # For animal detection: cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
    # Note: YOLO v8 has some limitations with animal detection, custom models may work better
    
    "yolov8n.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "hash": None,
        "size_mb": 6,
        "description": "YOLOv8 Nano for animal detection (fastest, lowest quality)"
    },
    "yolov8s.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
        "hash": None,
        "size_mb": 22,
        "description": "YOLOv8 Small for animal detection (balanced)"
    },
    "yolov8m.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
        "hash": None,
        "size_mb": 49,
        "description": "YOLOv8 Medium for animal detection"
    },
    "yolov8l.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
        "hash": None,
        "size_mb": 83,
        "description": "YOLOv8 Large for animal detection (high quality)"
    },
    "yolov8x.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt",
        "hash": None,
        "size_mb": 140,
        "description": "YOLOv8 Extra Large for animal detection (best quality)"
    },
}


def get_animal_pose_models_dir():
    """
    Get the ComfyUI animal pose models directory path.
    
    Returns path following ComfyUI structure:
    ComfyUI/models/animal_pose/
    """
    try:
        import folder_paths
        models_dir = os.path.join(folder_paths.models_dir, "animal_pose")
        os.makedirs(models_dir, exist_ok=True)
        return models_dir
    except ImportError:
        # Fallback if ComfyUI is not available
        fallback_dir = os.path.join(os.path.expanduser("~"), "ComfyUI", "models", "animal_pose")
        os.makedirs(fallback_dir, exist_ok=True)
        return fallback_dir


def register_animal_pose_models_folder():
    """Register the animal_pose models folder with ComfyUI's folder_paths."""
    try:
        import folder_paths
        models_dir = get_animal_pose_models_dir()
        folder_paths.add_model_folder_path("animal_pose", models_dir)
        logger.info(f"Registered animal_pose folder: {models_dir}")
    except ImportError:
        logger.warning("ComfyUI folder_paths not available - using fallback paths")
    except Exception as e:
        logger.warning(f"Could not register animal_pose folder: {e}")


def calculate_file_hash(filepath, algorithm="sha256"):
    """Calculate hash of a file for integrity verification."""
    hash_func = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def verify_model_file(filepath, expected_hash=None):
    """Verify that a model file exists and optionally check its hash."""
    if not os.path.exists(filepath):
        return False, "File does not exist"
    
    if expected_hash is None:
        return True, "File exists (hash not verified)"
    
    actual_hash = calculate_file_hash(filepath)
    if actual_hash.lower() == expected_hash.lower():
        return True, f"Hash verified: {actual_hash[:8]}..."
    else:
        return False, f"Hash mismatch! Expected: {expected_hash[:8]}..., Got: {actual_hash[:8]}..."


def download_model(model_name, progress_callback=None):
    """
    Download a model from the registry to the animal_pose models directory.
    
    Args:
        model_name (str): Name of the model to download
        progress_callback (callable): Optional callback for progress updates with args (downloaded_bytes, total_bytes)
    
    Returns:
        tuple: (success: bool, filepath: str, message: str)
    """
    if model_name not in ANIMAL_POSE_MODELS:
        return False, None, f"Model '{model_name}' not found in registry"
    
    model_info = ANIMAL_POSE_MODELS[model_name]
    url = model_info["url"]
    expected_hash = model_info.get("hash")
    
    models_dir = get_animal_pose_models_dir()
    filepath = os.path.join(models_dir, model_name)
    
    # Check if file already exists and is valid
    if os.path.exists(filepath):
        valid, msg = verify_model_file(filepath, expected_hash)
        if valid:
            logger.info(f"Model '{model_name}' already exists and is valid: {msg}")
            return True, filepath, f"Model already exists: {msg}"
        else:
            logger.warning(f"Existing model file is invalid: {msg}. Re-downloading...")
            os.remove(filepath)
    
    # Download the model
    logger.info(f"Downloading {model_name} from {url}...")
    
    try:
        def download_progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if progress_callback:
                progress_callback(downloaded, total_size)
        
        urllib.request.urlretrieve(url, filepath, reporthook=download_progress_hook)
        
        # Verify downloaded file
        if expected_hash:
            valid, msg = verify_model_file(filepath, expected_hash)
            if not valid:
                os.remove(filepath)
                return False, None, f"Downloaded file verification failed: {msg}"
            logger.info(f"Download verified: {msg}")
        
        logger.info(f"Successfully downloaded {model_name} to {filepath}")
        return True, filepath, f"Downloaded successfully to {filepath}"
    
    except Exception as e:
        error_msg = f"Failed to download {model_name}: {str(e)}"
        logger.error(error_msg)
        if os.path.exists(filepath):
            os.remove(filepath)
        return False, None, error_msg


def ensure_model_available(model_name, progress_callback=None):
    """
    Ensure a model is available locally, downloading if necessary.
    
    Args:
        model_name (str): Name of the model to ensure is available
        progress_callback (callable): Optional callback for progress updates
    
    Returns:
        tuple: (success: bool, filepath: str, message: str)
    """
    models_dir = get_animal_pose_models_dir()
    filepath = os.path.join(models_dir, model_name)
    
    # Check if model already exists
    if os.path.exists(filepath):
        expected_hash = ANIMAL_POSE_MODELS.get(model_name, {}).get("hash")
        valid, msg = verify_model_file(filepath, expected_hash)
        if valid:
            return True, filepath, f"Model already available: {msg}"
    
    # Download if not available
    return download_model(model_name, progress_callback)


def get_available_models():
    """Get list of models available for download."""
    return {name: info for name, info in ANIMAL_POSE_MODELS.items()}


def list_available_models():
    """Print list of available models with descriptions."""
    logger.info("Available Animal Pose Models:")
    logger.info("-" * 80)
    for model_name, info in ANIMAL_POSE_MODELS.items():
        logger.info(f"\n{model_name}")
        logger.info(f"  Description: {info['description']}")
        logger.info(f"  Size: ~{info['size_mb']} MB")
        logger.info(f"  URL: {info['url']}")
