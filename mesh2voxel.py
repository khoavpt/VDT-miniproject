import numpy as np
import open3d as o3d

# Function to load and voxelize a .ply mesh
def mesh_to_voxel(ply_file_path, voxel_size=0.05):
    # Step 1: Load the .ply mesh
    print("Loading .ply mesh file...")
    mesh = o3d.io.read_triangle_mesh(ply_file_path)
    
    # Check if the mesh is loaded correctly
    if not mesh.has_vertices():
        raise ValueError("Mesh file could not be loaded or contains no vertices.")
    
    # Step 2: Optional - Normalize the mesh to fit within a unit cube (helps with voxel size consistency)
    mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())
    
    # Step 3: Voxelize the mesh
    print(f"Voxelizing mesh with voxel size {voxel_size}...")
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)
    
    return mesh, voxel_grid

# Function to visualize the mesh and voxel grid
def visualize_mesh_and_voxel(mesh, voxel_grid):
    print("Visualizing original mesh and voxelized result...")
    # Set colors for better distinction
    mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Gray for mesh
    voxel_grid_color = [1, 0, 0]  # Red for voxels
    
    # Visualize both
    o3d.visualization.draw_geometries([mesh, voxel_grid])

# Function to save voxel grid as a point cloud (optional, for further use)
def save_voxel_as_point_cloud(voxel_grid, output_file="voxel_output.ply"):
    print(f"Saving voxel grid as point cloud to {output_file}...")
    # Extract voxel centers as points
    voxel_centers = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    # Convert grid indices to physical coordinates
    points = voxel_grid.origin + voxel_centers * voxel_grid.voxel_size
    
    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Save to .ply file
    o3d.io.write_point_cloud(output_file, pcd)
    print(f"Saved voxel centers to {output_file}")

# Main execution
if __name__ == "__main__":
    # Specify your .ply file path here
    ply_file = "/home/canhdx/workspace/OpenLRM/outputs/meshes/3b04b0a0dc1244d28d99382f7d33d54e.ply"  # Replace with your actual file path
    
    # Parameters
    voxel_size = 0.005  # Adjust this based on your mesh size (smaller = finer resolution)
    
    try:
        # Convert mesh to voxel
        original_mesh, voxel_grid = mesh_to_voxel(ply_file, voxel_size)
        
        # Visualize the result
        visualize_mesh_and_voxel(original_mesh, voxel_grid)
        
        # Optional: Save the voxel grid as a point cloud
        save_voxel_as_point_cloud(voxel_grid, "voxelized_output.ply")
        
    except Exception as e:
        print(f"An error occurred: {e}")
