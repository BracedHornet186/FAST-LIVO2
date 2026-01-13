import open3d as o3d

def visualize_pcd(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # Replace with your pcd file path
    visualize_pcd("all_raw_points.pcd")
