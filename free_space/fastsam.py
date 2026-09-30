import cv2
import numpy as np
import torch
from ultralytics import FastSAM

from free_space.utils import calculate_iou


class FastSAMSeg:
    """Runs FastSAM ("segment everything") and uses the RWPS plane mask to pick out
    which segments are water. Also returns the segment contours, which the stixel
    step uses as obstacle edge cues.
    """

    def __init__(self, model_path: str = './weights/FastSAM-x.pt'):
        try:
            self.model = FastSAM(model_path)
        except Exception as e:
            raise RuntimeError(f"Error loading FastSAM model from {model_path}. Reason: {e}")

    def _segment_img(self, img: np.array, device: str = 'cuda'):
        results = self.model(img, device=device, retina_masks=True, verbose=False, half=True, show=False)
        return results[0]

    def get_contours_and_water_mask(self, img: np.array, input_mask: np.array, device: str = 'cuda', min_area=3000, iou_threshold=0.005):
        """
        Returns (contour_mask, upper_contour_mask, water_mask).

        Every segment overlapping `input_mask` (the RWPS plane mask) with IoU >=
        `iou_threshold` is accepted as water; the water mask is the union of accepted
        segments minus the rejected ones. If none pass, the best-overlapping segment
        is used. water_mask is None when input_mask is None.
        """
        H, W = img.shape[:2]
        contour_mask = np.zeros((H, W))
        result = self._segment_img(img, device=device)

        if result is None or not hasattr(result, 'masks') or result.masks is None:
            print("No masks found by fastSAM!")
            return contour_mask, None

        masks = result.masks.data

        _, H_mask, W_mask = masks.shape

        binary_masks = (masks > 0.5).to(torch.uint8)
        areas = binary_masks.view(binary_masks.size(0), -1).sum(dim=1)  # Sum over H*W
        valid_indices = areas > min_area
        binary_masks = binary_masks[valid_indices]
        binary_masks = binary_masks.cpu().numpy()

        contour_mask_intermediate = np.zeros((H_mask, W_mask), dtype=np.uint8)
        upper_contour_mask_intermediate = np.zeros((H_mask, W_mask), dtype=np.uint8)

        scale_w = W / W_mask
        scale_h = H / H_mask

        accepted_masks = []
        rejected_masks = []
        best_iou = - 1
        matched_index = - 1

        if input_mask is not None:
            input_mask_resized = cv2.resize(input_mask, (W_mask, H_mask), interpolation=cv2.INTER_NEAREST)

        # Process each mask: compute IoU with input_mask and classify mask as accepted or rejected.
        for i, mask in enumerate(binary_masks):
            if input_mask is not None:
                iou = calculate_iou(input_mask_resized, mask)
                if iou > best_iou:
                    best_iou = iou
                    matched_index = i
                if iou >= iou_threshold:
                    accepted_masks.append(mask)
                else:
                    rejected_masks.append(mask)
            else:
                accepted_masks.append(mask)

            # Draw contours regardless of IoU.
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) >= min_area:
                    cv2.drawContours(contour_mask_intermediate, [contour], -1, 255, thickness=1)

            upper_line = self.get_upper_contour_line(mask)
            cv2.bitwise_or(upper_contour_mask_intermediate, upper_line, upper_contour_mask_intermediate)

        # If no masks were accepted, use the best mask found
        if input_mask is not None and not accepted_masks and matched_index != -1:
            print("using best mask")
            print(f"Best IoU: {best_iou} at index {matched_index}")
            accepted_masks.append(binary_masks[matched_index])
            rejected_masks = [
                mask for j, mask in enumerate(binary_masks) if j != matched_index
            ]

        contour_mask_resized = cv2.resize(contour_mask_intermediate, None, fx=scale_w, fy=scale_h, interpolation=cv2.INTER_NEAREST)
        upper_contour_mask_resized = cv2.resize(upper_contour_mask_intermediate, None, fx=scale_w, fy=scale_h, interpolation=cv2.INTER_NEAREST)

        cleaned_mask_resized = None
        if input_mask is not None and accepted_masks:
            accepted_combined = np.any(np.stack(accepted_masks), axis=0).astype(np.uint8)

            if rejected_masks:
                # Subtract regions of rejected masks from accepted union
                rejected_combined = np.any(np.stack(rejected_masks), axis=0).astype(np.uint8)
                cleaned_mask = accepted_combined & (~rejected_combined)
            else:
                cleaned_mask = accepted_combined

            cleaned_mask_resized = cv2.resize(cleaned_mask, (W, H), interpolation=cv2.INTER_NEAREST)

        return contour_mask_resized, upper_contour_mask_resized, cleaned_mask_resized

    def get_upper_contour_line(self, mask) -> np.array:
        """Mask with only the topmost pixel of `mask` set in each column."""
        has_nonzero = mask.any(axis=0)  # shape: (W,)

        # For a binary mask, argmax gives the first nonzero pixel. Columns with all
        # zeros also return 0, so only use columns that have a nonzero pixel.
        first_nonzero_indices = np.argmax(mask, axis=0)  # shape: (W,)

        result_mask = np.zeros_like(mask, dtype=np.uint8)

        valid_cols = np.where(has_nonzero)[0]
        result_mask[first_nonzero_indices[valid_cols], valid_cols] = 255

        return result_mask
