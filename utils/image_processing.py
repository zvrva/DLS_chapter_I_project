import numpy as np
import cv2
import tifffile as tiff
from torchvision import transforms

def load_tiff_image(path, target_size=384):
    try:
        image = tiff.imread(path)
        
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        elif image.shape[2] == 1:
            image = np.stack([image.squeeze()] * 3, axis=-1)
        
        original_shape = image.shape[:2]
        
        h, w = original_shape
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return image_resized.astype(np.float32) / 255.0, original_shape, (new_h, new_w)
    
    except Exception as e:
        raise Exception(f"Ошибка при загрузке файла: {e}")

def preprocess_image(image_array, image_size=384):
    image = cv2.resize(image_array, (image_size, image_size), interpolation=cv2.INTER_AREA)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def postprocess_mask(mask_pred, original_shape, processed_shape, threshold=0.5):
    mask_binary = (mask_pred > threshold).astype(np.uint8)
    
    mask_resized = cv2.resize(mask_binary, (processed_shape[1], processed_shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
    
    mask_final = cv2.resize(mask_resized, (original_shape[1], original_shape[0]), 
                           interpolation=cv2.INTER_NEAREST)
    
    return mask_final