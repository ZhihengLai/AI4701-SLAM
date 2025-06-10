# perform feature matching here

# return the matching result

import cv2
import numpy as np
import os
from tqdm import tqdm

def match_features(features):
    """
    匹配图像特征
    参数:
        features: 包含关键点和描述符的列表
    返回:
        matches: 匹配结果列表
    """
    # 使用FLANN匹配器，更适合SIFT特征
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = []
    
    # 创建输出目录
    os.makedirs("output", exist_ok=True)
    
    # 匹配所有相邻图像对
    for i in tqdm(range(len(features)-1)):
        des1 = features[i]['des']
        des2 = features[i+1]['des']
        
        if des1 is None or des2 is None:
            print(f"Warning: No descriptors for image {i} or {i+1}")
            matches.append([])
            continue
        
        # 应用比率测试的knn匹配
        try:
            knn_matches = flann.knnMatch(des1, des2, k=2)
        except Exception as e:
            print(f"Error during matching images {i} and {i+1}: {e}")
            matches.append([])
            continue
        
        # 应用Lowe's比率测试
        good_matches = []
        for m, n in knn_matches:
            if m.distance < 0.7 * n.distance:  # 比率阈值
                good_matches.append(m)
        
        # 基于RANSAC进一步过滤匹配
        if len(good_matches) >= 4:
            src_pts = np.float32([features[i]['kp'][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([features[i+1]['kp'][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is not None and mask is not None:
                ransac_matches = [good_matches[j] for j in range(len(good_matches)) if mask[j][0] == 1]
                good_matches = ransac_matches
        
        matches.append(good_matches)
        print(f"Found {len(good_matches)} good matches between images {i} and {i+1}")
        
        # Draw and save match visualization
        img_matches = cv2.drawMatches(features[i]['img'], features[i]['kp'], 
                                      features[i+1]['img'], features[i+1]['kp'], 
                                      good_matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(f"output/matches_{i}.jpg", img_matches)
    
    return matches