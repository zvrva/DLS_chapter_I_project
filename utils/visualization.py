import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import zipfile
import io

def create_overlay(original_image, mask, alpha=0.5):
    if original_image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), 
                         interpolation=cv2.INTER_NEAREST)
    
    if len(original_image.shape) == 3 and original_image.shape[2] == 3:
        colored_mask = np.zeros_like(original_image)
        colored_mask[mask > 0] = [255, 0, 0]
    else:
        if len(original_image.shape) == 2 or original_image.shape[2] == 1:
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        else:
            original_image_rgb = original_image
        
        colored_mask = np.zeros_like(original_image_rgb)
        colored_mask[mask > 0] = [255, 0, 0]
        original_image = original_image_rgb
    
    if original_image.dtype != np.uint8:
        original_uint8 = (original_image).astype(np.uint8)
    else:
        original_uint8 = original_image
    
    overlay = cv2.addWeighted(original_uint8, 1 - alpha, colored_mask, alpha, 0)
    return overlay

def create_results_zip(original_image, mask, overlay):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        original_pil = Image.fromarray((original_image * 255).astype(np.uint8))
        original_bytes = io.BytesIO()
        original_pil.save(original_bytes, format='PNG')
        zip_file.writestr('original_image.png', original_bytes.getvalue())
        
        mask_pil = Image.fromarray(mask * 255)
        mask_bytes = io.BytesIO()
        mask_pil.save(mask_bytes, format='PNG')
        zip_file.writestr('segmentation_mask.png', mask_bytes.getvalue())
        
        overlay_pil = Image.fromarray(overlay)
        overlay_bytes = io.BytesIO()
        overlay_pil.save(overlay_bytes, format='PNG')
        zip_file.writestr('segmentation_overlay.png', overlay_bytes.getvalue())
    
    return zip_buffer.getvalue()

def display_results(original_image, final_mask, overlay, threshold, alpha):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.imshow(final_mask, cmap='gray')
    ax1.axis('off')
    ax1.set_title(f"Маска сегментации (порог: {threshold})")
    
    ax2.imshow(overlay)
    ax2.axis('off')
    ax2.set_title(f"Наложение (прозрачность: {alpha})")
    
    return fig