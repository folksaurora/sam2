import os
import cv2
import argparse
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor


# ----------------------------
# UI: SELECT BBOX
# ----------------------------
def select_bbox(image):
    print("[INFO] Select object ROI and press ENTER/SPACE")
    bbox = cv2.selectROI("Select Object", image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Object")

    x, y, w, h = bbox
    if w == 0 or h == 0:
        raise RuntimeError("Invalid bbox selected")

    return np.array([x, y, x + w, y + h], dtype=np.float32)


# ----------------------------
# ARGS
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser("SAM2 video → COLMAP masks")

    parser.add_argument("--frames", required=True)
    parser.add_argument("--masks", required=True)

    parser.add_argument(
        "--checkpoint",
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..',
                         "checkpoints/sam2.1_hiera_base_plus.pt")
        )
    )

    parser.add_argument(
        "--config",
        default="configs/sam2.1/sam2.1_hiera_b+.yaml"
    )

    parser.add_argument("--viz_video", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--obj_id", type=int, default=1)

    return parser.parse_args()


# ----------------------------
# HELPERS
# ----------------------------
def mask_to_2d(mask_tensor):
    mask = mask_tensor.detach().cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]
    return mask


def save_mask(mask_tensor, out_path, target_size=None):
    mask = mask_to_2d(mask_tensor)

    mask = (mask > 0).astype(np.uint8) * 255
    mask = np.ascontiguousarray(mask)

    if target_size is not None:
        h, w = mask.shape
        th, tw = target_size
        if (h, w) != (th, tw):
            mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(out_path, mask)
    return True


# ----------------------------
# MAIN
# ----------------------------
def main():
    args = parse_args()

    frames_dir = args.frames
    masks_dir = args.masks
    os.makedirs(masks_dir, exist_ok=True)

    frame_names = [
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    def extract_number(f):
        name = os.path.splitext(f)[0]
        try:
            return int(name)
        except ValueError:
            return name

    frame_names.sort(key=extract_number)
    frame_paths = [os.path.join(frames_dir, f) for f in frame_names]

    total_frames = len(frame_paths)
    print(f"[INFO] Found {total_frames} frames")

    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise RuntimeError("Failed to load first frame")

    H, W = first_frame.shape[:2]
    target_size = (H, W)

    bbox = select_bbox(first_frame)
    print(f"[INFO] bbox = {bbox.tolist()}")

    # ----------------------------
    # VIDEO WRITER
    # ----------------------------
    video_writer = None
    if args.viz_video is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            args.viz_video,
            fourcc,
            10,
            (W * 2, H)
        )
        print(f"[INFO] Writing video → {args.viz_video}")

    # ----------------------------
    # MODEL
    # ----------------------------
    predictor = build_sam2_video_predictor(
        args.config,
        args.checkpoint,
        device=args.device
    )

    # ----------------------------
    # INFERENCE
    # ----------------------------
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):

        state = predictor.init_state(video_path=frames_dir)
        predictor.reset_state(state)

        x0, y0, x1, y1 = bbox

        points = np.array([
            [(x0 + x1) / 2, (y0 + y1) / 2],
            [x0, y0],
        ], dtype=np.float32)

        labels = np.array([1, 0], np.int32)

        predictor.add_new_points_or_box(
            state,
            frame_idx=0,
            obj_id=args.obj_id,
            points=points,
            labels=labels,
            box=bbox
        )

        mid_frame = total_frames // 2

        predictor.add_new_points_or_box(
            state,
            frame_idx=mid_frame,
            obj_id=args.obj_id,
            box=bbox
        )

        print("[INFO] Propagating...")

        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):

            frame = cv2.imread(frame_paths[frame_idx])
            mask_tensor = mask_logits[0]

            out_name = os.path.splitext(frame_names[frame_idx])[0] + ".png"
            out_path = os.path.join(masks_dir, out_name)

            # ----------------------------
            # SAVE MASK
            # ----------------------------
            save_mask(mask_tensor, out_path, target_size)

            # ----------------------------
            # VISUALIZATION
            # ----------------------------
            if video_writer is not None:

                mask_np = mask_to_2d(mask_tensor)
                mask_bin = (mask_np > 0)

                overlay = frame.copy()
                overlay[~mask_bin] = (0, 0, 255)

                viz = np.hstack([frame, overlay])
                video_writer.write(viz)

    if video_writer is not None:
        video_writer.release()
        print("[INFO] Video saved.")

    print("Done.")


if __name__ == "__main__":
    main()