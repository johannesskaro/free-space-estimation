import cv2
import numpy as np
from ultralytics import YOLO


class YoloSeg:
    """YOLO instance segmentation, used only to find boats (COCO class 8)."""

    def __init__(self, model_path: str = './weights/yolo11n-seg.pt'):
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(f"Error loading YOLO model from {model_path}. Reason: {e}")

    def get_boat_mask(self, img: np.array, device: str = 'cuda') -> np.array:
        (H, W, D) = img.shape

        results = self.model.predict(
            img,
            device=device,
            show=False,
            retina_masks=False,
            classes=[8],
            iou=0.5,
            verbose=False,
            half=True
        )

        r = results[0].masks

        boat_mask = np.zeros((H, W), dtype=np.uint8)

        if hasattr(r, 'xy'):
            masks = r.xy
            for mask in masks:
                polygon = np.round(mask.reshape((-1, 1, 2))).astype(np.int32)

                cv2.fillPoly(boat_mask, [polygon], color=255)

        return boat_mask

    def refine_water_mask(self, boat_mask, water_mask):
        """Remove boat pixels from the water mask."""
        boat_mask = (boat_mask > 0).astype(np.uint8)
        water_mask = water_mask.astype(np.uint8)
        inverted_boat_mask = cv2.bitwise_not(boat_mask)
        refined_water = cv2.bitwise_and(water_mask, inverted_boat_mask)
        return refined_water
