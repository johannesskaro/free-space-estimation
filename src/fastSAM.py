import numpy as np
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPredictor
import cv2
import torch
import torch.nn.functional as F
from utilities import calculate_iou

class FastSAMSeg:
    """
    A class to handle FastSAM segmentation tasks.
    """

    def __init__(self, model_path: str ='./weights/FastSAM-x.pt'):
        """
        Initialize the FastSAMSeg class.

        Parameters:
        - model_path (str): Path to the pretrained FastSAM model.
        """
        try:
            self.model = FastSAM(model_path)
        except Exception as e:
            raise RuntimeError(f"Error loading FastSAM model from {model_path}. Reason: {e}")
        
    def _segment_img(self, img: np.array, device: str = 'cuda') -> np.array:
        """
        Internal method to perform segmentation on the provided image.

        Parameters:
        - img (np.array): Input image for segmentation.
        - device (str): Device to run the model on, e.g., 'cuda'.

        Returns:
        - np.array: Segmentation results.
        """
        retina_masks = False
        verbose = False
        half = True
        imgsz = 640
        results = self.model(img, device=device, retina_masks=retina_masks, verbose=verbose, half=half, imgsz=imgsz)

        return results[0]

    def get_mask_at_points(self, img: np.array, points: np.array, pointlabel: np.array, device: str = 'cuda') -> np.array:
        """
        Obtain masks for specific points on the image.

        Parameters:
        - img (np.array): Input image.
        - points (np.array): Array of points.
        - pointlabel (np.array): Corresponding labels for points.
        - device (str): Device to run the model on, e.g., 'cuda'.

        Returns:
        - np.array: Mask result.
        """
        mask = np.zeros((img.shape[0], img.shape[1]))
        point_results = self.model(img, points=points, labels=pointlabel, device=device, retina_masks=True, verbose=False)
        ann = point_results[0].cpu()
        if len(ann)>0:
            mask = np.array(ann[0].masks.data[0]) if ann[0].masks else np.zeros((img.shape[0], img.shape[1]))

        return mask
    
    def get_mask_at_bbox(self, img: np.array, bbox: np.array, device: str = 'mps') -> np.array:
        """
        Obtain masks for the bounding box on the image.

        Parameters:
        - img (np.array): Input image.
        - bbox (np.array): Bounding box.
        - device (str): Device to run the model on, e.g., 'cuda'.

        Returns:
        - np.array: Mask result.
        """
        box_results = self.model(img, bboxes=bbox, device=device, retina_masks=True)
        ann = box_results[0].cpu()
        mask = np.array(ann[0].masks.data[0]) if ann[0].masks else np.zeros((img.shape[0], img.shape[1]))

        return mask
    
    def get_all_masks(self, img: np.array, device: str = 'cuda', min_area=3000) -> np.array:
        """
        Obtain all masks for the input image.

        Parameters:
        - img (np.array): Input image.
        - device (str): Device to run the model on, e.g., 'cuda'.

        Returns:
        - np.array: Masks result.
        """

        results = self._segment_img(img, device=device)
        if results[0].masks is not None:
            masks = np.array(results[0].masks.data.cpu())
        else: 
            masks = np.array([])

        valid_masks = []
        for mask in masks:
            # Convert to a binary mask
            binary_mask = (mask > 0).astype(np.uint8)
            # Optionally multiply by 255 if needed: binary_mask = binary_mask * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check if any contour meets the minimum area requirement
            for contour in contours:
                if cv2.contourArea(contour) >= min_area:
                    valid_masks.append(mask)
                    break  # Stop checking further contours for this mask

        valid_masks = np.array(valid_masks)
        return masks
    
    def get_all_masks_fast(self, img: np.array, device: str = 'cuda') -> np.array:
        results = self._segment_img(img, device=device)


        return results
    
    def get_all_countours(self, img: np.array, device: str = 'cuda', min_area=1000) -> np.array:
        """
        Obtain all contours for the input image.

        Parameters:
        - img (np.array): Input image.
        - device (str): Device to run the model on, e.g., 'cuda'.

        Returns:
        - np.array: Contours result.
        """
        H, W = img.shape[:2]
        #H, W = 1080, 1920

        contour_mask = np.zeros((H, W))

        result = self._segment_img(img, device=device)

        if result is None or not hasattr(result, 'masks') or result.masks is None:
            print("No masks found!")
            return contour_mask 

        masks = result.masks.data

        N, H_mask, W_mask = masks.shape

        binary_masks = (masks > 0.5).to(torch.uint8)
        areas = binary_masks.view(binary_masks.size(0), -1).sum(dim=1)  # Sum over H*W
        valid_indices = areas > min_area
        binary_masks = binary_masks[valid_indices]

        binary_masks = binary_masks.cpu().numpy()

        contour_mask_intermediate = np.zeros((H_mask, W_mask), dtype=np.uint8)
        
        for mask in binary_masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) >= min_area:
                    cv2.drawContours(contour_mask_intermediate, [contour], -1, 255, thickness=1)

        scale_w = W / W_mask
        scale_h = H / H_mask

        contour_mask_resized = cv2.resize(contour_mask_intermediate, None, fx=scale_w, fy=scale_h, interpolation=cv2.INTER_NEAREST)

        return contour_mask_resized
    
    def get_all_countours_and_best_iou_mask(self, img: np.array, input_mask: np.array, device: str = 'cuda', min_area=1000) -> np.array:

        H, W = img.shape[:2]
        contour_mask = np.zeros((H, W))

        result = self._segment_img(img, device=device)

        if result is None or not hasattr(result, 'masks') or result.masks is None:
            print("No masks found by fastSAM!")
            return contour_mask, None

        masks = result.masks.data

        N, H_mask, W_mask = masks.shape

        binary_masks = (masks > 0.5).to(torch.uint8)
        areas = binary_masks.view(binary_masks.size(0), -1).sum(dim=1)  # Sum over H*W
        valid_indices = areas > min_area
        binary_masks = binary_masks[valid_indices]
        binary_masks = binary_masks.cpu().numpy()
        
        contour_mask_intermediate = np.zeros((H_mask, W_mask), dtype=np.uint8)

        scale_w = W / W_mask
        scale_h = H / H_mask

        best_iou = 0
        matched_index = -1

        if input_mask is not None:
            input_mask_resized = cv2.resize(input_mask, (W_mask, H_mask), interpolation=cv2.INTER_NEAREST)
        
        for i, mask in enumerate(binary_masks):
            if input_mask is not None:
                iou = calculate_iou(input_mask_resized, mask)
                if iou > best_iou:
                    best_iou = iou
                    matched_index = i

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) >= min_area:
                    cv2.drawContours(contour_mask_intermediate, [contour], -1, 255, thickness=1)

        contour_mask_resized = cv2.resize(contour_mask_intermediate, None, fx=scale_w, fy=scale_h, interpolation=cv2.INTER_NEAREST)

        if input_mask is not None and matched_index >= 0:
            matched_mask = binary_masks[matched_index]

            # Build combined mask of all others
            all_other_masks = [
                mask for i, mask in enumerate(binary_masks) if i != matched_index
            ]
            if all_other_masks:
                combined_other_masks = np.any(np.stack(all_other_masks), axis=0).astype(np.uint8)
                cleaned_mask = matched_mask & (~combined_other_masks)
            else:
                cleaned_mask = matched_mask

            # Final resize
            cleaned_mask_resized = cv2.resize(cleaned_mask, (W, H), interpolation=cv2.INTER_NEAREST)

            return contour_mask_resized, cleaned_mask_resized

        # Fall back to returning only the contour mask
        return contour_mask_resized, None
    


    
    def get_all_upper_countours(self, img: np.array, device: str = 'cuda', min_area=3000) -> np.array:
        masks = self.get_all_masks(img, device=device)
        combined_upper = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        for mask in masks:
            # Ensure binary mask
            bin_mask = (mask > 0).astype(np.uint8)
            # Optionally, you can use cv2.findContours to filter out small regions
            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = False
            for contour in contours:
                if cv2.contourArea(contour) >= min_area:
                    valid = True
                    break
            if not valid:
                continue

            # Extract the upper contour line from this mask.
            upper_line = self.get_upper_contour_line(bin_mask)
            # Optionally, you can also draw the original contour:
            # cv2.drawContours(upper_line, [contour], -1, 255, thickness=1)

            # Combine the upper contour line with previous ones.
            combined_upper = cv2.bitwise_or(combined_upper, upper_line)

        return combined_upper
    
    def get_upper_contour_line(self, mask) -> np.array:
        has_nonzero = mask.any(axis=0)  # shape: (W,)
        
        # For each column, np.argmax returns the index of the first occurrence of the maximum value.
        # For a binary mask (0 or 1), this gives the first nonzero pixel. Note that for columns
        # with all zeros, np.argmax returns 0 even though no pixel is nonzero.
        first_nonzero_indices = np.argmax(mask, axis=0)  # shape: (W,)
        
        # Create the result mask (initialize with zeros)
        result_mask = np.zeros_like(mask, dtype=np.uint8)
        
        # Only update the columns that have at least one nonzero pixel.
        valid_cols = np.where(has_nonzero)[0]
        result_mask[first_nonzero_indices[valid_cols], valid_cols] = 255

        return result_mask
    

