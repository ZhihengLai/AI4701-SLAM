# perform 3D reconstruction using PnP

import cv2
import numpy as np
from tqdm import tqdm

def pnp_reconstruction(features, matches, K, initial_points, initial_poses):
    """
    使用PnP进行后续图像的三维重建
    参数:
        features: 特征列表
        matches: 匹配列表
        K: 相机内参矩阵
        initial_points: 初始三维点
        initial_poses: 初始相机位姿
    返回:
        all_points: 所有三维点
        all_poses: 所有相机位姿
    """
    all_poses = initial_poses.copy()
    all_points = initial_points.copy()
    
    # 创建点索引映射，记录已重建的3D点
    point_index_map = {}  # 键: (图像索引, 特征点索引), 值: 3D点索引
    
    # 初始化前两帧的点索引映射
    for i in tqdm(range(len(matches[0]))):
        match = matches[0][i]
        if i < len(initial_points):  # 确保不超出初始点云的范围
            point_index_map[(0, match.queryIdx)] = i
            point_index_map[(1, match.trainIdx)] = i
    
    for i in tqdm(range(1, len(features)-1)):
        # 获取当前图像和下一幅图像的特征和匹配
        kp_curr = features[i]['kp']
        kp_next = features[i+1]['kp']
        curr_matches = matches[i]
        
        # 收集已知3D点和对应的2D点用于PnP
        obj_points = []  # 3D点
        img_points = []  # 对应的2D点
        match_indices = []  # 保存匹配的索引
        
        for j, match in enumerate(curr_matches):
            # 检查当前帧中的点是否已经有对应的3D点
            if (i, match.queryIdx) in point_index_map:
                # 找到已重建的3D点索引
                point_3d_idx = point_index_map[(i, match.queryIdx)]
                # 确保索引有效
                if point_3d_idx < len(all_points):
                    obj_points.append(all_points[point_3d_idx])
                    img_points.append(kp_next[match.trainIdx].pt)
                    match_indices.append(j)
        
        if len(obj_points) < 4:
            print(f"Warning: Not enough points for PnP in image {i+1}. Only {len(obj_points)} points found.")
            continue
        
        obj_points = np.array(obj_points, dtype=np.float32)
        img_points = np.array(img_points, dtype=np.float32)
        
        # 使用PnP求解相机位姿
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_points, img_points, K, None, confidence=0.99, reprojectionError=8.0)
        
        if not success or inliers is None or len(inliers) < 4:
            print(f"PnP failed for image {i+1}. Skipping.")
            continue
        
        # 将旋转向量转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvec)
        
        # 构建相机位姿矩阵
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = tvec.reshape(3)
        all_poses.append(pose)
        
        # 三角测量新的三维点
        P1 = K @ np.hstack((all_poses[i][:3, :3], all_poses[i][:3, [3]]))
        P2 = K @ np.hstack((R, tvec))
        
        # 获取匹配点用于三角测量
        pts1 = np.float32([kp_curr[m.queryIdx].pt for m in curr_matches]).reshape(-1, 2)
        pts2 = np.float32([kp_next[m.trainIdx].pt for m in curr_matches]).reshape(-1, 2)
        
        # 三角测量所有匹配点
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = (points_4d[:3] / points_4d[3]).T
        
        # 筛选有效的三角测量点 (在相机前方且重投影误差小)
        valid_points = []
        valid_indices = []
        
        for j, pt3d in enumerate(points_3d):
            # 检查点是否在两个相机前方
            if pt3d[2] <= 0:
                continue
                
            # 计算重投影点
            proj1, _ = cv2.projectPoints(pt3d.reshape(1, 3), np.zeros(3), np.zeros(3), K, None)
            proj2, _ = cv2.projectPoints(pt3d.reshape(1, 3), rvec, tvec, K, None)
            
            # 计算重投影误差
            error1 = np.linalg.norm(proj1.ravel() - pts1[j])
            error2 = np.linalg.norm(proj2.ravel() - pts2[j])
            
            if error1 < 5.0 and error2 < 5.0:  # 阈值可调整
                valid_points.append(pt3d)
                valid_indices.append(j)
                # 更新点索引映射
                new_idx = len(all_points)
                point_index_map[(i, curr_matches[j].queryIdx)] = new_idx
                point_index_map[(i+1, curr_matches[j].trainIdx)] = new_idx
        
        if valid_points:
            valid_points = np.array(valid_points)
            all_points = np.vstack((all_points, valid_points))
            print(f"PnP reconstruction: added {len(valid_points)} new points. Total: {len(all_points)}")
        else:
            print(f"Warning: No valid points triangulated for image {i+1}")
    
    return all_points, all_poses