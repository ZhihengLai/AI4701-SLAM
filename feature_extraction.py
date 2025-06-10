# perform feature extraction here
# return the feature vector

import cv2
import numpy as np
from tqdm import tqdm
import os # Import os for path joining

def resize_image_for_storage(image, max_dimension=1024):
    """
    Resizes an image to have its largest dimension be max_dimension, preserving aspect ratio.
    This is to reduce memory footprint when storing many images.
    """
    h, w = image.shape[:2]
    
    if h == 0 or w == 0: # Should not happen with valid images
        return image

    if max(h, w) <= max_dimension:
        return image.copy() # Return a copy if no resize needed, to ensure it's a new object if original is modified

    if h > w:
        ratio = max_dimension / float(h)
        new_h = max_dimension
        new_w = int(w * ratio)
    else:
        ratio = max_dimension / float(w)
        new_w = max_dimension
        new_h = int(h * ratio)
        
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def extract_features(image_folder, image_filenames): # Modified signature
    """
    使用SIFT算法提取图像特征
    参数:
        image_folder: 存放图像的文件夹路径
        image_filenames: 图像文件名列表
    返回:
        features: 包含关键点和描述符的列表. 
                  'img' field will contain a resized image for memory efficiency.
    """
    sift = cv2.SIFT_create()
    features = []
    
    for idx, filename in tqdm(enumerate(image_filenames), desc="Extracting Features", total=len(image_filenames)):
        img_path = os.path.join(image_folder, filename)
        img_original = cv2.imread(img_path)

        if img_original is None:
            print(f"Warning: Could not load image {filename}. Skipping.")
            # Append a placeholder or handle missing images if necessary for downstream consistency
            # For now, just skip and the features list will be shorter than image_filenames
            features.append({'img': None, 'kp': None, 'des': None, 'filename': filename}) # Store filename for reference
            continue
            
        # 转换为灰度图 (full resolution)
        gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        # 检测关键点并计算描述符 (on full resolution gray image)
        kp, des = sift.detectAndCompute(gray, None)
        
        # 保存带有关键点的图像以供中间结果查看 (on full resolution original image)
        # Consider making this step optional or resizing before saving if disk space/time is an issue
        # For now, keeping as is, as it's a requirement.
        img_kp_display = cv2.drawKeypoints(img_original, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(f"output/features_{idx}_{filename}.jpg", img_kp_display) # Include filename for clarity
        
        # Resize image for storage in the features list to save memory
        img_resized_for_storage = resize_image_for_storage(img_original, max_dimension=1024) # Adjust max_dimension as needed
        
        features.append({'img': img_resized_for_storage, 'kp': kp, 'des': des, 'filename': filename})
        
        # img_original, gray, img_kp_display will go out of scope and be eligible for garbage collection
    
    return features