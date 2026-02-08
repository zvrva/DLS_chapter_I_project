import numpy as np
import torch
import torchvision


def load_car_detector(weights_path: str, device: torch.device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)

    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    elif isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        model.load_state_dict(state)
    else:
        raise ValueError("Не удалось определить формат весов для модели детекции.")

    model.to(device)
    model.eval()
    return model


def _tile_to_tensor(rgb_u8: np.ndarray) -> torch.Tensor:
    if rgb_u8.ndim == 2:
        rgb_u8 = np.stack([rgb_u8] * 3, axis=-1)
    return torch.from_numpy(rgb_u8).permute(2, 0, 1).float() / 255.0


@torch.no_grad()
def scan_car_bboxes(
    full_image_u8: np.ndarray,
    model,
    device: torch.device,
    tile_size: int = 384,
    stride: int = 384,
    score_thr: float = 0.8,
    target_cars: int | None = None,
    progress_cb=None,
):
    h, w = full_image_u8.shape[:2]
    total_tiles = ((h + stride - 1) // stride) * ((w + stride - 1) // stride)

    all_boxes = []
    tiles_done = 0
    found_any = False
    found_count = 0

    y = 0
    while y < h:
        x = 0
        while x < w:
            tile = full_image_u8[y:min(y + tile_size, h), x:min(x + tile_size, w)]
            tile_t = _tile_to_tensor(tile).to(device)

            out = model([tile_t])[0]
            boxes = out["boxes"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()
            keep = scores >= score_thr
            boxes = boxes[keep]

            if len(boxes) > 0:
                found_any = True
                boxes_full = boxes.copy()
                boxes_full[:, [0, 2]] += x
                boxes_full[:, [1, 3]] += y
                all_boxes.append(boxes_full)
                found_count += boxes_full.shape[0]

            tiles_done += 1
            if progress_cb is not None:
                progress_cb(tiles_done, total_tiles, found_count)

            if found_any and target_cars is not None:
                if found_count >= target_cars:
                    break

            x += stride
        if found_any and target_cars is not None:
            if found_count >= target_cars:
                break
        y += stride

    if len(all_boxes) > 0:
        all_boxes = np.concatenate(all_boxes, axis=0)
    else:
        all_boxes = np.zeros((0, 4), dtype=np.float32)

    if target_cars is not None and len(all_boxes) > target_cars:
        all_boxes = all_boxes[:target_cars]

    return all_boxes, tiles_done, (w, h)
