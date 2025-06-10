# perform bundle adjustment here

import cv2
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from tqdm import tqdm

def bundle_adjustment(points_3d, poses, features, matches, K):
    """
    执行光束法平差优化
    参数:
        points_3d: 三维点云
        poses: 相机位姿列表
        features: 特征列表
        matches: 匹配列表
        K: 相机内参矩阵
    返回:
        optimized_points: 优化后的三维点
        optimized_poses: 优化后的相机位姿
    """
    n_cameras = len(poses)
    n_points = points_3d.shape[0]
    
    # 构建观测数据
    point_indices = []
    camera_indices = []
    points_2d = []
    
    # 创建点索引映射，用于跟踪在不同视图中看到的3D点
    point_map = {}  # (img_idx, kp_idx) -> 3d_point_idx
    
    # 处理初始匹配
    for i in tqdm(range(min(n_cameras-1, len(matches)))):
        match_list = matches[i]
        for m in match_list:
            # 为简单起见，假设前两张图像的匹配点对应的3D点索引与匹配点索引相同
            # 实际应用中需要更复杂的跟踪机制
            if i == 0:  # 第一对图像
                point_idx = len(point_map)
                point_map[(0, m.queryIdx)] = point_idx
                point_map[(1, m.trainIdx)] = point_idx
                
                # 添加到观测数据
                point_indices.append(point_idx)
                camera_indices.append(0)
                points_2d.append(features[0]['kp'][m.queryIdx].pt)
                
                point_indices.append(point_idx)
                camera_indices.append(1)
                points_2d.append(features[1]['kp'][m.trainIdx].pt)
            else:
                # 对于后续图像，检查当前点是否已经有对应的3D点
                if (i, m.queryIdx) in point_map:
                    point_idx = point_map[(i, m.queryIdx)]
                    # 为下一帧添加观测
                    point_map[(i+1, m.trainIdx)] = point_idx
                    
                    point_indices.append(point_idx)
                    camera_indices.append(i+1)
                    points_2d.append(features[i+1]['kp'][m.trainIdx].pt)
    
    # 转换为numpy数组
    point_indices = np.array(point_indices)
    camera_indices = np.array(camera_indices)
    points_2d = np.array(points_2d)
    
    # 确保索引不超出范围
    max_point_idx = np.max(point_indices) if len(point_indices) > 0 else 0
    if max_point_idx >= n_points:
        print(f"Warning: Point indices exceed n_points ({max_point_idx} >= {n_points})")
        # 调整n_points以适应所有点
        n_points = max_point_idx + 1
        if n_points > points_3d.shape[0]:
            # 扩展points_3d数组
            additional_points = np.zeros((n_points - points_3d.shape[0], 3))
            points_3d = np.vstack([points_3d, additional_points])
    
    n_observations = len(points_2d)
    print(f"Bundle adjustment: {n_cameras} cameras, {n_points} points, {n_observations} observations")
    
    # 准备优化参数
    camera_params = []
    for pose in poses:
        # 将旋转矩阵转换为旋转向量
        rvec, _ = cv2.Rodrigues(pose[:3, :3])
        camera_params.extend(rvec.ravel())
        camera_params.extend(pose[:3, 3].ravel())
    
    # 构建参数向量
    x0 = np.hstack([np.array(camera_params), points_3d.ravel()])
    
    # 构建稀疏雅可比矩阵的结构
    A = lil_matrix((2 * n_observations, 6 * n_cameras + 3 * n_points), dtype=np.float32)
    
    # 填充雅可比矩阵结构
    i = np.arange(n_observations)
    for s in range(6):
        A[2 * i, 6 * camera_indices + s] = 1
        A[2 * i + 1, 6 * camera_indices + s] = 1
    
    for s in range(3):
        A[2 * i, 6 * n_cameras + 3 * point_indices + s] = 1
        A[2 * i + 1, 6 * n_cameras + 3 * point_indices + s] = 1
    
    # 定义优化函数
    def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
        """计算重投影误差"""
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))
        
        projected = project(points_3d[point_indices], camera_params[camera_indices], K)
        
        # 计算残差 (flatten 后的重投影误差)
        return (projected - points_2d).ravel()
    
    # 定义投影函数
    def project(points, camera_params, K):
        """将3D点投影到图像平面"""
        n_points = points.shape[0]
        projections = np.zeros((n_points, 2))
        
        for i in range(n_points):
            # 旋转向量和平移向量
            rvec = camera_params[i, :3]
            tvec = camera_params[i, 3:6]
            
            # 投影单个点
            point = points[i].reshape(1, 3)
            proj, _ = cv2.projectPoints(point, rvec, tvec, K, None)
            projections[i] = proj.reshape(2)
            
        return projections
    
    # 如果没有足够的观测，跳过优化
    if n_observations < 10:
        print("Not enough observations for bundle adjustment")
        return points_3d, poses
    
    # 执行优化
    try:
        res = least_squares(
            fun, x0, 
            jac_sparsity=A, 
            verbose=2, 
            x_scale='jac',
            ftol=1e-4, 
            method='trf',
            args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K),
            max_nfev=100)
        
        # 提取优化后的参数
        optimized_params = res.x
        optimized_camera_params = optimized_params[:n_cameras * 6].reshape((n_cameras, 6))
        optimized_points = optimized_params[n_cameras * 6:].reshape((n_points, 3))
        
        # 转换回相机位姿矩阵
        optimized_poses = []
        for i in range(n_cameras):
            rvec = optimized_camera_params[i, :3]
            tvec = optimized_camera_params[i, 3:6]
            
            R, _ = cv2.Rodrigues(rvec)
            
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = tvec
            optimized_poses.append(pose)
            
        print("Bundle adjustment completed successfully")
        return optimized_points, optimized_poses
        
    except Exception as e:
        print(f"Bundle adjustment failed: {e}")
        return points_3d, poses