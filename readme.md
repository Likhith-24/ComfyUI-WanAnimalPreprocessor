# ComfyUI-WanAnimalPreprocess

ComfyUI custom nodes for **animal pose estimation** using [ViTPose](https://github.com/JunkyByte/easy_ViTPose) models and YOLOv8 detection. Supports both **AP10k** and **APT36k** datasets (17 keypoints each). Adapted from [kijai/ComfyUI-WanAnimatePreprocess](https://github.com/kijai/ComfyUI-WanAnimatePreprocess) (human pose) for animal use.

## Features

- **Animal detection** via YOLOv8 ONNX — cats, dogs, horses, sheep, cows, elephants, bears, zebras, giraffes, birds
- **17-keypoint animal pose estimation** via ViTPose ONNX
- **Dual dataset support** — AP10k and APT36k, selectable per workflow
- **Pose retargeting** from template video to reference animal image
- **OneToAll animation** integration with ref/pose/none alignment modes
- Skeleton visualization with configurable stick width, head toggle

## AP10k / APT36k Keypoints (17)

Both datasets use the same 17-keypoint skeleton:

```
 0: L_Eye         5: L_Shoulder   11: L_Hip
 1: R_Eye         6: L_Elbow      12: L_Knee
 2: Nose          7: L_F_Paw      13: L_B_Paw
 3: Neck          8: R_Shoulder   14: R_Hip
 4: Root_of_tail  9: R_Elbow      15: R_Knee
                  10: R_F_Paw     16: R_B_Paw
```

### AP10k vs APT36k

| | AP10k | APT36k |
|---|---|---|
| **Images** | ~10,000 | ~36,000 |
| **Species** | 23 animal families | 30 species |
| **Best for** | Common animals (cat, dog, horse) | Broader species coverage |
| **Keypoints** | 17 (identical format) | 17 (identical format) |

Use **AP10k** for typical domestic/farm animals. Use **APT36k** for wider species variety or when AP10k underperforms on your specific animal. Both model types are interchangeable in this node — just select the matching dataset in the model loader.

## Installation

1. Clone into `ComfyUI/custom_nodes/`:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/your-username/ComfyUI-WanAnimalPreprocess.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Models

Place all ONNX models in `ComfyUI/models/detection/`.

### ViTPose (Pose Estimation)

Download ONNX models from [JunkyByte/easy_ViTPose on HuggingFace](https://huggingface.co/JunkyByte/easy_ViTPose):

**AP10k models:**

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `vitpose-s-ap10k.onnx` | ~45 MB | Fastest | Good |
| `vitpose-b-ap10k.onnx` | ~90 MB | Fast | Better |
| `vitpose-l-ap10k.onnx` | ~150 MB | Medium | High |
| `vitpose-h-ap10k.onnx` | ~300 MB | Slow | Best |

**APT36k models:**

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `vitpose-s-apt36k.onnx` | ~45 MB | Fastest | Good |
| `vitpose-b-apt36k.onnx` | ~90 MB | Fast | Better |
| `vitpose-l-apt36k.onnx` | ~150 MB | Medium | High |
| `vitpose-h-apt36k.onnx` | ~300 MB | Slow | Best |

### YOLOv8 (Animal Detection)

Any YOLOv8 ONNX model works. The nodes automatically filter for COCO animal classes (bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe).

Recommended: `yolov8m.onnx` or `yolov8l.onnx` for good balance of speed and accuracy.

## Nodes

### ONNX Animal Detection Model Loader
Loads ViTPose and YOLO ONNX models.

**Inputs:**
- `vitpose_model` — ViTPose ONNX file from detection folder
- `yolo_model` — YOLOv8 ONNX file from detection folder
- `dataset` — **ap10k** or **apt36k** (must match your ViTPose model)
- `onnx_device` — CUDA or CPU

**Output:** `POSEMODEL` — model bundle for downstream nodes

### Animal Pose and Detection
Main processing node for image/video pose estimation.

**Inputs:**
- `model` — from model loader
- `images` — input frames (IMAGE batch)
- `width` / `height` — target output dimensions
- `retarget_image` (optional) — reference image for pose retargeting

**Outputs:**
- `pose_data` — full pose pipeline data (POSEDATA)
- `key_frame_body_points` — JSON string of key body point coordinates
- `bboxes` — detected animal bounding boxes

### Draw Animal ViTPose
Renders pose skeleton images from pose data.

**Inputs:**
- `pose_data` — from detection node
- `width` / `height` — canvas dimensions
- `retarget_padding` — padding for retarget resize (0 = disabled)
- `body_stick_width` — skeleton line width (-1 = auto)
- `draw_head` — toggle head keypoints (eyes, nose)

**Output:** `pose_images` — rendered skeleton frames (IMAGE batch)

### Animal Pose Retarget Prompt Helper
Generates text prompts describing the detected animal pose (e.g., "All four legs and paws are visible").

### Animal Pose Detection OneToAll Animation
Full OneToAll animation pipeline with pose alignment.

**Inputs:**
- `model` — from model loader
- `images` — driving video frames
- `width` / `height` — output dimensions
- `align_to` — **ref** (retarget to reference), **pose** (warp reference to pose), **none**
- `draw_head` — **full**, **weak**, or **none**
- `ref_image` (optional) — reference animal image

**Outputs:**
- `pose_images` — aligned pose skeleton frames
- `ref_pose_image` — reference pose visualization
- `ref_image` — processed reference image
- `ref_mask` — reference mask

## Workflow

A basic workflow:

1. **Load Models** → ONNX Animal Detection Model Loader (select dataset: ap10k or apt36k)
2. **Detect Poses** → Animal Pose and Detection (feed images + optional retarget reference)
3. **Draw Skeletons** → Draw Animal ViTPose (render pose images)
4. **Generate** → Feed pose images into WanAnimate / your generation pipeline

For OneToAll animation, use the dedicated `Animal Pose Detection OneToAll Animation` node instead of steps 2-3.

## Credits

- [kijai/ComfyUI-WanAnimatePreprocess](https://github.com/kijai/ComfyUI-WanAnimatePreprocess) — Original human pose ComfyUI nodes
- [JunkyByte/easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose) — ViTPose AP10k/APT36k models and ONNX export
- [Alibaba Wan Team](https://github.com/Wan-Video/Wan2.1) — WanAnimate framework
- [ssj9596/One-to-All-Animation](https://github.com/ssj9596/One-to-All-Animation) — OneToAll animation support
- [AP10k Dataset](https://github.com/AlexTheBad/AP-10K) — Animal Pose estimation benchmark
- [APT36k Dataset](https://github.com/pandorgan/APT-36K) — Animal Pose Tracking benchmark

## License

See [LICENSE](LICENSE) file.
