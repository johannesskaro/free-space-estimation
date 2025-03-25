from ultralytics import YOLO
import cv2
import numpy as np


class YoloSeg:

    def __init__(self, model_path: str ='./weights/yolo11n-seg'):
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(f"Error loading YOLO model from {model_path}. Reason: {e}")
        
    def get_boat_mask(self, img: np.array, device: str = 'cuda') -> np.array:
        (H, W, D) = img.shape

        results = self.model.predict(
            img, device=device, 
            show=False,
            retina_masks=True, 
            classes=[8], 
            iou=0.5, 
            verbose=False
        )

        r = results[0].masks

        overlay = img.copy()
        
        boat_mask = np.zeros((H, W), dtype=np.uint8)

        if hasattr(r, 'xy'):
            masks = r.xy 
            for mask in masks:
                polygon = np.round(mask.reshape((-1, 1, 2))).astype(np.int32)

                cv2.fillPoly(boat_mask, [polygon], color=255)

        #cv2.imshow("Boat mask", boat_mask)

        return boat_mask
    
    def refine_water_mask(self, boat_mask, water_mask):
        boat_mask = (boat_mask > 0).astype(np.uint8)
        water_mask = water_mask.astype(np.uint8)
        inverted_boat_mask = cv2.bitwise_not(boat_mask)
        refined_water = cv2.bitwise_and(water_mask, inverted_boat_mask)
        return refined_water



def plot_yolo_masks(image, results):
    """
    Plots all segmentation masks from a YOLO Model in different colors using OpenCV.

    :param image: Original image (H, W, 3) in NumPy format.
    :param masks: Binary masks (N, H, W), where N is the number of detected objects.
    """
    
    # Ensure the image is in RGB format
    if len(image.shape) == 2 or image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Create a copy of the image to draw on
    overlay = image.copy()

    # Overlay each mask in a different color
    r = results[0].masks
    if hasattr(r, 'xy'):
        masks = r.xy  # or simply r.masks depending on your API

        for mask in masks:
            mask = np.round(mask.reshape((-1, 1, 2))).astype(np.int32)
            cv2.polylines(overlay, [mask], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imshow("YOLO Model - Mask Visualization", overlay)