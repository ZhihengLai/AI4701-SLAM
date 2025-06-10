# use epipolar geometry to find the fundamental matrix and the essential matrix
# use the essential matrix to find the relative pose between two cameras
# use the relative pose to triangulate 3D points

import cv2
import numpy as np
from tqdm import tqdm

def initial_reconstruction(features, matches, K):
    """
    使用对极几何进行初始重建
    参数:
        features: 特征列表
        matches: 匹配列表
        K: 相机内参矩阵
    返回:
        points_3d: 三维点云
        poses: 相机位姿列表
    """
    # 获取前两幅图像的特征点和匹配
    kp1 = features[0]['kp']
    kp2 = features[1]['kp']
    match = matches[0]
    
    if len(match) < 8:
        raise ValueError("Not enough matches for initial reconstruction")
    
    # 获取匹配点的坐标
    pts1 = np.float32([kp1[m.queryIdx].pt for m in match])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in match])
    
    # 计算基础矩阵
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
    
    if F is None or mask is None or np.sum(mask) < 8:
        raise ValueError("Could not compute a valid fundamental matrix")
    
    # 保留内点
    mask = mask.ravel().astype(bool)
    pts1 = pts1[mask]
    pts2 = pts2[mask]
    
    # 从匹配中提取内点
    inlier_matches = [m for i, m in enumerate(match) if mask[i]]
    
    # 计算本质矩阵
    E = K.T @ F @ K
    
    # 从本质矩阵恢复相机位姿
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    
    # 第一个相机位姿设置为单位矩阵
    pose1 = np.eye(4)
    # 第二个相机位姿
    pose2 = np.eye(4)
    pose2[:3, :3] = R
    pose2[:3, 3] = t.reshape(3)
    
    # 构建投影矩阵
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    
    # 三角测量获取3D点
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = points_4d[:3] / points_4d[3]
    points_3d = points_3d.T
    
    # 筛选有效的三维点（在两个相机前方且重投影误差小）
    valid_points = []
    valid_indices = []
    
    for i, pt3d in tqdm(enumerate(points_3d)):
        # 转换到相机坐标系
        pt_cam1 = pt3d.copy()  # 已经在第一个相机坐标系
        pt_cam2 = R @ pt3d + t.ravel()
        
        # 检查点是否在两个相机前方
        if pt_cam1[2] <= 0 or pt_cam2[2] <= 0:
            continue
            
        # 计算重投影点
        proj1, _ = cv2.projectPoints(pt3d.reshape(1, 3), np.zeros(3), np.zeros(3), K, None)
        proj2, _ = cv2.projectPoints(pt3d.reshape(1, 3), cv2.Rodrigues(R)[0], t, K, None)
        
        # 计算重投影误差
        error1 = np.linalg.norm(proj1.ravel() - pts1[i])
        error2 = np.linalg.norm(proj2.ravel() - pts2[i])
        
        if error1 < 5.0 and error2 < 5.0:  # 阈值可调整
            valid_points.append(pt3d)
            valid_indices.append(i)
    
    if len(valid_points) < 10:
        raise ValueError(f"Too few valid 3D points: {len(valid_points)}")
        
    valid_points = np.array(valid_points)
    print(f"Initial reconstruction: {len(valid_points)} valid points out of {len(points_3d)}")
    
    return valid_points, [pose1, pose2]