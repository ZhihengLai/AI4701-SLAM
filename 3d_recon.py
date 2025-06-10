# perform complete 3D reconstruction from images

import cv2
import numpy as np
import os
from tqdm import tqdm
from feature_extraction import extract_features
from feature_matching import match_features
from initial_recon import initial_reconstruction
from pnp_recon import pnp_reconstruction
from bundle_adjustment import bundle_adjustment
import open3d as o3d

def load_images(folder_path):
    """
    加载所有图像文件名
    """
    # images = [] # No longer load all images into memory here
    filenames = []
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Image folder not found: {folder_path}")
    
    file_list = sorted(os.listdir(folder_path))
    
    if not file_list:
        raise ValueError(f"No files found in folder: {folder_path}")
    
    for filename in tqdm(file_list, desc="Scanning Images"):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # img_path = os.path.join(folder_path, filename) # Path construction will be in extract_features
            # img = cv2.imread(img_path) # Don't read here
            
            # if img is None:
            #     print(f"Warning: Could not load image {filename}") # This check will move
            #     continue
                
            # images.append(img) # Don't append image data
            filenames.append(filename)
    
    if not filenames:
        raise ValueError("No valid image filenames found in the specified folder")
        
    print(f"Found {len(filenames)} image files in {folder_path}")
    return filenames # Return only filenames

def save_results(points_3d, poses, output_folder):
    """
    保存重建结果
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # 保存点云为PLY文件
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    # 计算点云颜色 (这里简单设为灰度)
    colors = np.ones((len(points_3d), 3)) * 0.5
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    ply_path = os.path.join(output_folder, "reconstruction.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Point cloud saved to {ply_path} ({len(points_3d)} points)")
    
    # 保存相机位姿为TXT文件
    pose_path = os.path.join(output_folder, "camera_poses.txt")
    with open(pose_path, 'w') as f:
        for i, pose in enumerate(poses):
            # 确保pose是4x4矩阵，如果不是，则补全
            if pose.shape == (3, 4):
                full_pose = np.vstack((pose, np.array([0, 0, 0, 1])))
            elif pose.shape == (4, 4):
                full_pose = pose
            else:
                print(f"Warning: Pose {i} has unexpected shape {pose.shape}. Skipping.")
                continue
            
            flattened = full_pose.ravel() # 拉平为16个浮点数
            line = ' '.join(map(str, flattened))
            f.write(line + '\n')
    print(f"Camera poses saved to {pose_path} ({len(poses)} poses)")

def load_camera_intrinsics(file_path):
    """
    从文件加载相机内参矩阵
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Camera intrinsics file not found: {file_path}")
        
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 3:
            raise ValueError("Camera intrinsics file must contain at least 3 lines")
            
        mat = [list(map(float, line.split())) for line in lines if line.strip()]
        intrinsics = np.array(mat)
        
        if intrinsics.shape != (3, 3):
            raise ValueError(f"Expected 3x3 intrinsics matrix, got {intrinsics.shape}")
            
        return intrinsics
    except Exception as e:
        raise ValueError(f"Failed to load camera intrinsics: {e}")

def main():
    try:
        # 参数设置
        image_folder = "images/images"  # 图像文件夹路径
        output_folder = "output"  # 输出文件夹路径
        intrinsics_file = "camera_intrinsic.txt"  # 相机内参文件
        
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)
        
        # 1. 加载相机内参
        print("Loading camera intrinsics...")
        K = load_camera_intrinsics(intrinsics_file)
        print("Camera intrinsics loaded successfully")
        
        # 2. 加载图像文件名
        print("Scanning image folder...")
        image_filenames = load_images(image_folder) # Now gets only filenames
        print(f"Found {len(image_filenames)} images to process.")
        
        # 3. 特征提取
        print("Extracting features...")
        # Pass image_folder and image_filenames to extract_features
        features = extract_features(image_folder, image_filenames) 
        print("Feature extraction completed")
        
        # 4. 特征匹配
        print("Matching features...")
        matches = match_features(features)
        print("Feature matching completed")
        
        # 5. 初始重建：使用前两张图
        print("Performing initial reconstruction...")
        points_3d, poses = initial_reconstruction(features, matches, K)
        print(f"Initial reconstruction completed with {len(points_3d)} points and {len(poses)} poses")
        
        # 6. PnP重建
        print("Performing PnP reconstruction...")
        points_3d, poses = pnp_reconstruction(features, matches, K, points_3d, poses)
        print(f"PnP reconstruction completed with {len(points_3d)} points and {len(poses)} poses")
        
        # 7. 光束法平差
        print("Performing bundle adjustment...")
        points_3d, poses = bundle_adjustment(points_3d, poses, features, matches, K)
        print(f"Bundle adjustment completed with {len(points_3d)} points and {len(poses)} poses")
        
        # 8. 保存结果
        print("Saving results...")
        save_results(points_3d, poses, output_folder)
        
        print("3D reconstruction completed successfully!")
        
    except Exception as e:
        print(f"Error during reconstruction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()