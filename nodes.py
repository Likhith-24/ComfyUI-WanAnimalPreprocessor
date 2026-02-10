# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified for animal pose estimation using AP10k/APT36k ViTPose models.
# Original: https://github.com/kijai/ComfyUI-WanAnimatePreprocess
# Animal models: https://github.com/JunkyByte/easy_ViTPose

import os
import torch
from tqdm import tqdm
import numpy as np
import folder_paths
import cv2
import json
import logging

script_directory = os.path.dirname(os.path.abspath(__file__))

from comfy import model_management as mm
from comfy.utils import ProgressBar

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

folder_paths.add_model_folder_path("detection", os.path.join(folder_paths.models_dir, "detection"))

from .models.onnx_models import ViTPose, Yolo
from .pose_utils.pose2d_utils import load_pose_metas_from_kp2ds_seq, crop, bbox_from_detector
from .utils import padding_resize, resize_by_area, resize_to_bounds
from .pose_utils.human_visualization import AAPoseMeta, draw_animal_pose_by_meta
from .retarget_pose import get_retarget_pose

# COCO class IDs (1-indexed) for animals detectable by YOLOv8
# bird(15), cat(16), dog(17), horse(18), sheep(19), cow(20),
# elephant(21), bear(22), zebra(23), giraffe(24)
ANIMAL_CAT_IDS = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]


class OnnxAnimalDetectionModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vitpose_model": (folder_paths.get_filename_list("detection"), {
                    "tooltip": "ViTPose ONNX model for animal pose estimation. "
                               "Loaded from 'ComfyUI/models/detection' folder.",
                }),
                "yolo_model": (folder_paths.get_filename_list("detection"), {
                    "tooltip": "YOLOv8 ONNX model for animal detection. "
                               "Loaded from 'ComfyUI/models/detection' folder.",
                }),
                "dataset": (["ap10k", "apt36k"], {
                    "default": "ap10k",
                    "tooltip": "Dataset the ViTPose model was trained on. "
                               "AP10k: 23 animal families, good for common animals (cat, dog, horse). "
                               "APT36k: 30 species, broader coverage and more training data.",
                }),
                "onnx_device": (["CUDAExecutionProvider", "CPUExecutionProvider"], {
                    "default": "CUDAExecutionProvider",
                    "tooltip": "Device to run the ONNX models on",
                }),
            },
        }

    RETURN_TYPES = ("POSEMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loadmodel"
    CATEGORY = "WanAnimalPreprocess"
    DESCRIPTION = "Loads ONNX models for animal pose detection. Supports both AP10k and APT36k datasets (both use 17 keypoints). Select the dataset matching your ViTPose model."

    def loadmodel(self, vitpose_model, yolo_model, dataset, onnx_device):
        vitpose_model_path = folder_paths.get_full_path_or_raise("detection", vitpose_model)
        yolo_model_path = folder_paths.get_full_path_or_raise("detection", yolo_model)

        vitpose = ViTPose(vitpose_model_path, onnx_device)
        yolo = Yolo(yolo_model_path, onnx_device, cat_id=ANIMAL_CAT_IDS, select_type='max')

        model = {
            "vitpose": vitpose,
            "yolo": yolo,
            "dataset": dataset,
        }

        return (model,)


class AnimalPoseAndDetection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("POSEMODEL",),
                "images": ("IMAGE",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 1,
                                   "tooltip": "Width of the generation"}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 1,
                                    "tooltip": "Height of the generation"}),
            },
            "optional": {
                "retarget_image": ("IMAGE", {"default": None,
                                              "tooltip": "Optional reference image for pose retargeting"}),
            },
        }

    RETURN_TYPES = ("POSEDATA", "STRING", "BBOX")
    RETURN_NAMES = ("pose_data", "key_frame_body_points", "bboxes")
    FUNCTION = "process"
    CATEGORY = "WanAnimalPreprocess"
    DESCRIPTION = "Detects animal poses from images using ViTPose (AP10k/APT36k) and YOLO. Optionally retargets poses based on a reference image."

    def process(self, model, images, width, height, retarget_image=None):
        detector = model["yolo"]
        pose_model = model["vitpose"]
        dataset = model.get("dataset", "ap10k")

        B, H, W, C = images.shape
        shape = np.array([H, W])[None]
        images_np = images.numpy()

        IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
        IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
        input_resolution = (256, 192)
        rescale = 1.25

        detector.reinit()
        pose_model.reinit()

        # Process optional retarget reference image
        refer_pose_meta = None
        refer_img = None
        if retarget_image is not None:
            refer_img = resize_by_area(retarget_image[0].numpy() * 255, width * height, divisor=16) / 255.0
            ref_bbox = (detector(
                cv2.resize(refer_img.astype(np.float32), (640, 640)).transpose(2, 0, 1)[None],
                shape
            )[0][0]["bbox"])

            if ref_bbox is None or ref_bbox[-1] <= 0 or (ref_bbox[2] - ref_bbox[0]) < 10 or (ref_bbox[3] - ref_bbox[1]) < 10:
                ref_bbox = np.array([0, 0, refer_img.shape[1], refer_img.shape[0]])

            center, scale = bbox_from_detector(ref_bbox, input_resolution, rescale=rescale)
            refer_img = crop(refer_img, center, scale, (input_resolution[0], input_resolution[1]))[0]

            img_norm = (refer_img - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

            ref_keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
            refer_pose_meta = load_pose_metas_from_kp2ds_seq(
                ref_keypoints, width=retarget_image.shape[2], height=retarget_image.shape[1]
            )[0]

        # Detect bounding boxes
        comfy_pbar = ProgressBar(B * 2)
        progress = 0
        bboxes = []
        for img in tqdm(images_np, total=len(images_np), desc="Detecting animal bboxes"):
            bboxes.append(detector(
                cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None],
                shape
            )[0][0]["bbox"])
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        detector.cleanup()

        # Extract keypoints
        kp2ds = []
        for img, bbox in tqdm(zip(images_np, bboxes), total=len(images_np), desc="Extracting animal keypoints"):
            if bbox is None or bbox[-1] <= 0 or (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 10:
                bbox = np.array([0, 0, img.shape[1], img.shape[0]])

            bbox_xywh = bbox
            center, scale = bbox_from_detector(bbox_xywh, input_resolution, rescale=rescale)
            img = crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]

            img_norm = (img - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

            keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
            kp2ds.append(keypoints)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_model.cleanup()

        kp2ds = np.concatenate(kp2ds, 0)
        pose_metas = load_pose_metas_from_kp2ds_seq(kp2ds, width=W, height=H)

        # Retarget or convert poses
        if retarget_image is not None and refer_pose_meta is not None:
            retarget_pose_metas = get_retarget_pose(pose_metas[0], refer_pose_meta, pose_metas, None, None)
        else:
            retarget_pose_metas = [AAPoseMeta.from_humanapi_meta(meta) for meta in pose_metas]

        bbox = np.array(bboxes[0]).flatten()
        if bbox.shape[0] >= 4:
            bbox_ints = tuple(int(v) for v in bbox[:4])
        else:
            bbox_ints = (0, 0, 0, 0)

        # Generate key frame body points for prompt/reference
        key_frame_num = 4 if B >= 4 else 1
        key_frame_step = len(pose_metas) // key_frame_num
        key_frame_index_list = list(range(0, len(pose_metas), key_frame_step))

        # AP10k key point indices: Nose(2), Neck(3), L_Shoulder(5), R_Shoulder(8), L_Hip(11), R_Hip(14)
        key_points_index = [2, 3, 5, 8, 11, 14]

        points_dict_list = []
        for key_frame_index in key_frame_index_list:
            keypoints_body_list = []
            body_key_points = pose_metas[key_frame_index]['keypoints_body']
            for each_index in key_points_index:
                each_keypoint = body_key_points[each_index]
                if each_keypoint is None:
                    continue
                keypoints_body_list.append(each_keypoint)

            if len(keypoints_body_list) > 0:
                keypoints_body = np.array(keypoints_body_list)[:, :2]
                wh = np.array([[pose_metas[0]['width'], pose_metas[0]['height']]])
                points = (keypoints_body * wh).astype(np.int32)
                for point in points:
                    points_dict_list.append({"x": int(point[0]), "y": int(point[1])})

        pose_data = {
            "retarget_image": refer_img if retarget_image is not None else None,
            "pose_metas": retarget_pose_metas,
            "refer_pose_meta": refer_pose_meta if retarget_image is not None else None,
            "pose_metas_original": pose_metas,
            "dataset": dataset,
        }

        return (pose_data, json.dumps(points_dict_list), [bbox_ints])


class DrawAnimalViTPose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 1,
                                   "tooltip": "Width of the generation"}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 1,
                                    "tooltip": "Height of the generation"}),
                "retarget_padding": ("INT", {"default": 16, "min": 0, "max": 512, "step": 1,
                                              "tooltip": "When > 0, the retargeted pose image is padded and resized to the target size"}),
                "body_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1,
                                              "tooltip": "Width of the body sticks. Set to 0 to disable body drawing, -1 for auto"}),
                "draw_head": ("BOOLEAN", {"default": True,
                                           "tooltip": "Whether to draw head keypoints (eyes, nose)"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_images",)
    FUNCTION = "process"
    CATEGORY = "WanAnimalPreprocess"
    DESCRIPTION = "Draws animal pose skeleton images from pose data (AP10k/APT36k format)."

    def process(self, pose_data, width, height, body_stick_width, draw_head, retarget_padding=64):
        retarget_image = pose_data.get("retarget_image", None)
        pose_metas = pose_data["pose_metas"]
        dataset = pose_data.get("dataset", "ap10k")

        use_retarget_resize = retarget_padding > 0 and retarget_image is not None

        comfy_pbar = ProgressBar(len(pose_metas))
        progress = 0
        crop_target_image = None
        pose_images = []

        for idx, meta in enumerate(tqdm(pose_metas, desc="Drawing animal pose images")):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            pose_image = draw_animal_pose_by_meta(
                canvas, meta,
                draw_head=draw_head,
                body_stick_width=body_stick_width,
                dataset=dataset,
            )

            if crop_target_image is None:
                crop_target_image = pose_image

            if use_retarget_resize:
                pose_image = resize_to_bounds(
                    pose_image, height, width,
                    crop_target_image=crop_target_image,
                    extra_padding=retarget_padding,
                )
            else:
                pose_image = padding_resize(pose_image, height, width)

            pose_images.append(pose_image)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_images_np = np.stack(pose_images, 0)
        pose_images_tensor = torch.from_numpy(pose_images_np).float() / 255.0

        return (pose_images_tensor,)


class AnimalPoseRetargetPromptHelper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "retarget_prompt")
    FUNCTION = "process"
    CATEGORY = "WanAnimalPreprocess"
    DESCRIPTION = "Generates text prompts for animal pose retargeting based on visibility of limbs in the template pose."

    def process(self, pose_data):
        refer_pose_meta = pose_data.get("refer_pose_meta", None)
        if refer_pose_meta is None:
            return ("Change the animal to face forward.", "Change the animal to face forward.")

        tpl_pose_metas = pose_data["pose_metas_original"]
        front_legs_visible = False
        back_legs_visible = False

        for tpl_pose_meta in tpl_pose_metas:
            tpl_keypoints = tpl_pose_meta['keypoints_body']
            tpl_keypoints = np.array(tpl_keypoints)

            # Check front legs: L_Shoulder(5), L_Elbow(6), L_F_Paw(7), R_Shoulder(8), R_Elbow(9), R_F_Paw(10)
            front_indices = [5, 6, 7, 8, 9, 10]
            for idx in front_indices:
                if (tpl_keypoints[idx][0] <= 1 and tpl_keypoints[idx][1] <= 1 and tpl_keypoints[idx][2] >= 0.75):
                    front_legs_visible = True
                    break

            # Check back legs: L_Hip(11), L_Knee(12), L_B_Paw(13), R_Hip(14), R_Knee(15), R_B_Paw(16)
            back_indices = [11, 12, 13, 14, 15, 16]
            for idx in back_indices:
                if (tpl_keypoints[idx][0] <= 1 and tpl_keypoints[idx][1] <= 1 and tpl_keypoints[idx][2] >= 0.75):
                    back_legs_visible = True
                    break

            if front_legs_visible and back_legs_visible:
                break

        if back_legs_visible and front_legs_visible:
            tpl_prompt = "Change the animal to a standard standing pose facing forward. All four legs and paws are visible."
            refer_prompt = "Change the animal to a standard standing pose facing forward. All four legs and paws are visible."
        elif front_legs_visible:
            tpl_prompt = "Change the animal to face forward. Front legs are visible."
            refer_prompt = "Change the animal to face forward. Front legs are visible."
        else:
            tpl_prompt = "Change the animal to face forward."
            refer_prompt = "Change the animal to face forward."

        return (tpl_prompt, refer_prompt)


class AnimalPoseDetectionOneToAllAnimation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("POSEMODEL",),
                "images": ("IMAGE",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 2,
                                   "tooltip": "Width of the generation"}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 2,
                                    "tooltip": "Height of the generation"}),
                "align_to": (["ref", "pose", "none"], {"default": "ref",
                                                        "tooltip": "Alignment mode for poses"}),
                "draw_head": (["full", "weak", "none"], {"default": "full",
                                                          "tooltip": "Whether to draw head keypoints on the pose images"}),
            },
            "optional": {
                "ref_image": ("IMAGE", {"default": None,
                                         "tooltip": "Optional reference image for pose retargeting"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("pose_images", "ref_pose_image", "ref_image", "ref_mask")
    FUNCTION = "process"
    CATEGORY = "WanAnimalPreprocess"
    DESCRIPTION = "Specialized animal pose detection and alignment for OneToAllAnimation model. Detects animal poses from input images and aligns them based on a reference image if provided."

    def process(self, model, images, width, height, align_to, draw_head, ref_image=None):
        from .onetoall.infer_function import aaposemeta_to_dwpose, align_to_reference, align_to_pose
        from .onetoall.utils import draw_pose_aligned, warp_ref_to_pose

        detector = model["yolo"]
        pose_model = model["vitpose"]
        dataset = model.get("dataset", "ap10k")
        B, H, W, C = images.shape

        shape = np.array([H, W])[None]
        images_np = images.numpy()

        IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
        IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
        input_resolution = (256, 192)
        rescale = 1.25

        detector.reinit()
        pose_model.reinit()

        # Process ref image if provided
        ref_dwpose = None
        refer_pose_meta = None
        refer_img_np = None
        if ref_image is not None:
            refer_img_np = ref_image[0].numpy() * 255
            refer_img = resize_by_area(refer_img_np, width * height, divisor=16) / 255.0
            ref_bbox = (detector(
                cv2.resize(refer_img.astype(np.float32), (640, 640)).transpose(2, 0, 1)[None],
                shape
            )[0][0]["bbox"])

            if ref_bbox is None or ref_bbox[-1] <= 0 or (ref_bbox[2] - ref_bbox[0]) < 10 or (ref_bbox[3] - ref_bbox[1]) < 10:
                ref_bbox = np.array([0, 0, refer_img.shape[1], refer_img.shape[0]])

            center, scale = bbox_from_detector(ref_bbox, input_resolution, rescale=rescale)
            refer_img = crop(refer_img, center, scale, (input_resolution[0], input_resolution[1]))[0]

            img_norm = (refer_img - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

            ref_keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
            refer_pose_meta = load_pose_metas_from_kp2ds_seq(
                ref_keypoints, width=ref_image.shape[2], height=ref_image.shape[1]
            )[0]

            ref_dwpose = aaposemeta_to_dwpose(refer_pose_meta)

        # Detect bboxes
        comfy_pbar = ProgressBar(B * 2)
        progress = 0
        bboxes = []
        for img in tqdm(images_np, total=len(images_np), desc="Detecting animal bboxes"):
            bboxes.append(detector(
                cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None],
                shape
            )[0][0]["bbox"])
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        detector.cleanup()

        # Extract keypoints
        kp2ds = []
        for img, bbox in tqdm(zip(images_np, bboxes), total=len(images_np), desc="Extracting animal keypoints"):
            if bbox is None or bbox[-1] <= 0 or (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 10:
                bbox = np.array([0, 0, img.shape[1], img.shape[0]])

            bbox_xywh = bbox
            center, scale = bbox_from_detector(bbox_xywh, input_resolution, rescale=rescale)
            img = crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]

            img_norm = (img - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

            keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
            kp2ds.append(keypoints)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_model.cleanup()

        kp2ds = np.concatenate(kp2ds, 0)
        pose_metas = load_pose_metas_from_kp2ds_seq(kp2ds, width=W, height=H)
        tpl_dwposes = [aaposemeta_to_dwpose(meta) for meta in pose_metas]

        # Process alignment
        ref_pose_image_tensor = None
        if ref_image is not None and ref_dwpose is not None:
            if align_to == "ref":
                ref_pose_image = draw_pose_aligned(ref_dwpose, height, width, without_face=True)
                ref_pose_image_np = np.stack(ref_pose_image, 0) if isinstance(ref_pose_image, list) else np.array(ref_pose_image)
                ref_pose_image_tensor = torch.from_numpy(ref_pose_image_np).unsqueeze(0).float() / 255.0
                tpl_dwposes = align_to_reference(refer_pose_meta, pose_metas, tpl_dwposes, anchor_idx=0)
                image_input_tensor = ref_image
                image_mask_tensor = torch.zeros(1, ref_image.shape[1], ref_image.shape[2], dtype=torch.float32, device="cpu")
            elif align_to == "pose":
                image_input, ref_pose_image_np, image_mask = warp_ref_to_pose(refer_img_np, tpl_dwposes[0], ref_dwpose)
                ref_pose_image_np = np.stack(ref_pose_image_np, 0) if isinstance(ref_pose_image_np, list) else np.array(ref_pose_image_np)
                ref_pose_image_tensor = torch.from_numpy(ref_pose_image_np).unsqueeze(0).float() / 255.0
                tpl_dwposes = align_to_pose(ref_dwpose, tpl_dwposes, anchor_idx=0)
                image_input_tensor = torch.from_numpy(image_input).unsqueeze(0).float() / 255.0
                image_mask_tensor = torch.from_numpy(image_mask).unsqueeze(0).float() / 255.0
            elif align_to == "none":
                ref_pose_image = draw_pose_aligned(ref_dwpose, height, width, without_face=True)
                ref_pose_image_np = np.stack(ref_pose_image, 0) if isinstance(ref_pose_image, list) else np.array(ref_pose_image)
                ref_pose_image_tensor = torch.from_numpy(ref_pose_image_np).unsqueeze(0).float() / 255.0
                image_input_tensor = ref_image
                image_mask_tensor = torch.zeros(1, ref_image.shape[1], ref_image.shape[2], dtype=torch.float32, device="cpu")
        else:
            ref_pose_image_tensor = torch.zeros((1, height, width, 3), dtype=torch.float32, device="cpu")
            image_input_tensor = torch.zeros((1, height, width, 3), dtype=torch.float32, device="cpu")
            image_mask_tensor = torch.zeros(1, height, width, dtype=torch.float32, device="cpu")

        # Draw pose images
        pose_imgs = []
        for pose_np in tpl_dwposes:
            pose_img = draw_pose_aligned(
                pose_np, height, width,
                without_face=True,
                head_strength=draw_head,
            )
            pose_img = torch.from_numpy(np.array(pose_img))
            pose_imgs.append(pose_img)

        pose_tensor = torch.stack(pose_imgs).cpu().float() / 255.0

        return (pose_tensor, ref_pose_image_tensor, image_input_tensor, image_mask_tensor)


NODE_CLASS_MAPPINGS = {
    "OnnxAnimalDetectionModelLoader": OnnxAnimalDetectionModelLoader,
    "AnimalPoseAndDetection": AnimalPoseAndDetection,
    "DrawAnimalViTPose": DrawAnimalViTPose,
    "AnimalPoseRetargetPromptHelper": AnimalPoseRetargetPromptHelper,
    "AnimalPoseDetectionOneToAllAnimation": AnimalPoseDetectionOneToAllAnimation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnnxAnimalDetectionModelLoader": "ONNX Animal Detection Model Loader",
    "AnimalPoseAndDetection": "Animal Pose and Detection",
    "DrawAnimalViTPose": "Draw Animal ViTPose",
    "AnimalPoseRetargetPromptHelper": "Animal Pose Retarget Prompt Helper",
    "AnimalPoseDetectionOneToAllAnimation": "Animal Pose Detection OneToAll Animation",
}
