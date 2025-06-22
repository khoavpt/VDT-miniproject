import torch
import trimesh
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
from kaolin.metrics.pointcloud import chamfer_distance, f_score
from typing import List, Tuple, Dict, Optional
import argparse

def load_mesh_as_pointcloud(file_path: str, num_points: int = 10000) -> torch.Tensor:
    """
    Load a mesh file and sample points from its surface.

    Args:
        file_path (str): Path to the mesh file.
        num_points (int): Number of points to sample from the surface.

    Returns:
        torch.Tensor: Sampled point cloud of shape (num_points, 3).
    """
    mesh = trimesh.load_mesh(file_path)
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return torch.tensor(points, dtype=torch.float32)

def normalize_and_align_mesh(pred_points: torch.Tensor, gt_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Center and normalize predicted and ground truth point clouds to the [-1, 1] range.

    Both point clouds are centered by subtracting their centroids and scaled independently
    to fit within a unit cube based on their maximum bounds.

    Args:
        pred_points (torch.Tensor): Predicted point cloud of shape (N, 3).
        gt_points (torch.Tensor): Ground truth point cloud of shape (N, 3).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, float, float]: Normalized predicted and ground truth point clouds,
        and their respective scale factors.
    """
    # Compute centroids
    pred_centroid = pred_points.mean(dim=0)  # Shape: (3,)
    gt_centroid = gt_points.mean(dim=0)      # Shape: (3,)

    # Center both point clouds at origin
    pred_points_centered = pred_points - pred_centroid
    gt_points_centered = gt_points - gt_centroid

    # Compute scaling factors to normalize to [-1, 1]
    pred_max_bound = torch.max(torch.abs(pred_points_centered))
    pred_scale_factor = 1.0 / pred_max_bound
    gt_max_bound = torch.max(torch.abs(gt_points_centered))
    gt_scale_factor = 1.0 / gt_max_bound

    print(f"Ground truth scale factor: {gt_scale_factor:.6f}")
    print(f"Predicted scale factor: {pred_scale_factor:.6f}")

    # Scale both point clouds
    pred_points_normalized = pred_points_centered * pred_scale_factor
    gt_points_normalized = gt_points_centered * gt_scale_factor

    return pred_points_normalized, gt_points_normalized, pred_scale_factor, gt_scale_factor

def fill_voxel_grid(mesh: o3d.geometry.TriangleMesh, voxel_size: float) -> o3d.geometry.VoxelGrid:
    """
    Convert a mesh into a filled voxel grid, including both surface and interior voxels.

    Uses ray casting to determine interior points based on signed distance.

    Args:
        mesh (o3d.geometry.TriangleMesh): Input mesh to voxelize.
        voxel_size (float): Size of each voxel.

    Returns:
        o3d.geometry.VoxelGrid: Voxel grid representing the filled mesh.
    """
    if not mesh.is_watertight():
        print("Warning: Mesh is not watertight, voxelization results may be inconsistent.")

    # Sample surface points
    pcd = mesh.sample_points_poisson_disk(number_of_points=10000)
    surface_points = np.asarray(pcd.points)

    # Define grid bounds with padding
    min_bound = mesh.get_min_bound() - voxel_size * 0.5
    max_bound = mesh.get_max_bound() + voxel_size * 0.5

    # Generate grid coordinates
    x = np.arange(min_bound[0], max_bound[0] + voxel_size, voxel_size)
    y = np.arange(min_bound[1], max_bound[1] + voxel_size, voxel_size)
    z = np.arange(min_bound[2], max_bound[2] + voxel_size, voxel_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1).astype(np.float32)

    # Perform ray casting to determine occupancy
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)
    signed_distance = scene.compute_signed_distance(grid_points).numpy()
    inside_points = grid_points[signed_distance < 0]  # Interior points

    # Combine surface and interior points
    all_points = np.vstack([surface_points, inside_points])
    filled_pcd = o3d.geometry.PointCloud()
    filled_pcd.points = o3d.utility.Vector3dVector(all_points)

    # Create voxel grid
    return o3d.geometry.VoxelGrid.create_from_point_cloud(filled_pcd, voxel_size)

def normalize_and_align_voxels(pred_mesh_path: str, gt_mesh_path: str, pred_scale_factor: float, 
                              gt_scale_factor: float, voxel_size: float = 0.025) -> Tuple[o3d.geometry.VoxelGrid, 
                                                                                          o3d.geometry.VoxelGrid, 
                                                                                          np.ndarray, np.ndarray]:
    """
    Load meshes, center them, apply scaling, and convert to voxel grids.

    The predicted mesh's Z-axis is flipped to align coordinate systems before scaling.

    Args:
        pred_mesh_path (str): Path to the predicted mesh file.
        gt_mesh_path (str): Path to the ground truth mesh file.
        pred_scale_factor (float): Scale factor for the predicted mesh from point cloud normalization.
        gt_scale_factor (float): Scale factor for the ground truth mesh from point cloud normalization.
        voxel_size (float): Size of each voxel.

    Returns:
        Tuple[o3d.geometry.VoxelGrid, o3d.geometry.VoxelGrid, np.ndarray, np.ndarray]: 
        Predicted and ground truth voxel grids, and their centroids.
    """
    # Load meshes
    pred_mesh = o3d.io.read_triangle_mesh(pred_mesh_path)
    gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)

    if not pred_mesh.has_vertices() or not gt_mesh.has_vertices():
        raise ValueError("One or both mesh files could not be loaded or contain no vertices.")

    # Center meshes at their centroids
    pred_centroid = pred_mesh.get_center()
    gt_centroid = gt_mesh.get_center()
    pred_mesh.translate(-pred_centroid)
    gt_mesh.translate(-gt_centroid)

    # Flip Z-axis of predicted mesh to align coordinate systems
    vertices = np.asarray(pred_mesh.vertices)
    # vertices = vertices[:, [2, 0, 1]]
    vertices[:, 1] = -vertices[:, 1]

    pred_mesh.vertices = o3d.utility.Vector3dVector(vertices)

    # Scale meshes using provided factors
    pred_mesh.scale(pred_scale_factor, center=(0, 0, 0))
    gt_mesh.scale(gt_scale_factor, center=(0, 0, 0))

    # Convert to voxel grids
    pred_voxel = fill_voxel_grid(pred_mesh, voxel_size)
    gt_voxel = fill_voxel_grid(gt_mesh, voxel_size)

    return pred_voxel, gt_voxel, pred_centroid, gt_centroid

def calculate_voxel_iou(voxel_grid1: o3d.geometry.VoxelGrid, voxel_grid2: o3d.geometry.VoxelGrid) -> float:
    """
    Calculate the Intersection over Union (IoU) between two voxel grids.

    Voxel grids are aligned by centering their voxel indices before computation.

    Args:
        voxel_grid1 (o3d.geometry.VoxelGrid): Ground truth voxel grid.
        voxel_grid2 (o3d.geometry.VoxelGrid): Predicted voxel grid.

    Returns:
        float: IoU value between 0 and 1.
    """
    voxels1 = np.array([v.grid_index for v in voxel_grid1.get_voxels()])
    voxels2 = np.array([v.grid_index for v in voxel_grid2.get_voxels()])

    if len(voxels1) == 0 or len(voxels2) == 0:
        return 0.0

    # Center voxel indices
    voxel1_centroid = np.mean(voxels1, axis=0)
    voxel2_centroid = np.mean(voxels2, axis=0)
    voxels1_centered = voxels1 - voxel1_centroid
    voxels2_centered = voxels2 - voxel2_centroid

    # Compute IoU using sets of rounded integer indices
    voxels1_set = set(tuple(v) for v in np.round(voxels1_centered).astype(int))
    voxels2_set = set(tuple(v) for v in np.round(voxels2_centered).astype(int))
    intersection = len(voxels1_set.intersection(voxels2_set))
    union = len(voxels1_set.union(voxels2_set))

    return intersection / union if union > 0 else 0.0

def evaluate_3d_models(gt_path: str, pred_path: str, plot_path: str, num_samples: int = 10000, 
                      thresholds: List[float] = [0.2, 0.3, 0.5], device: str = "cuda:0") -> Tuple[float, List[float], float]:
    """
    Evaluate 3D model reconstruction by comparing predicted and ground truth meshes.

    Computes Chamfer Distance, F-scores at specified thresholds, and voxel IoU.

    Args:
        gt_path (str): Path to ground truth mesh.
        pred_path (str): Path to predicted mesh.
        plot_path (str): Path to save visualization plots.
        num_samples (int): Number of points to sample from each mesh.
        thresholds (List[float]): Thresholds for F-score calculation.
        device (str): Device to perform computations on (e.g., "cuda:6").

    Returns:
        Tuple[float, List[float], float]: Chamfer Distance, list of F-scores, and voxel IoU.
    """
    # Load point clouds
    gt_points = load_mesh_as_pointcloud(gt_path, num_samples)
    pred_points = load_mesh_as_pointcloud(pred_path, num_samples)
    # pred_points = pred_points[:, [2, 0, 1]]  # Flip Z-axis to align with GT
    pred_points[:, 1] = -pred_points[:, 1]  # Flip Y-axis to align with GT

    if gt_points is None or pred_points is None:
        return None, None, None

    # Normalize and align point clouds
    pred_points_normalized, gt_points_normalized, pred_scale_factor, gt_scale_factor = normalize_and_align_mesh(pred_points, gt_points)

    # Plot normalized point clouds
    plot_point_clouds(gt_points_normalized, pred_points_normalized, plot_path, "Ground Truth vs Predicted Point Clouds (Normalized)")

    # Move to device for metric computation
    gt_points_normalized = gt_points_normalized.unsqueeze(0).to(device)
    pred_points_normalized = pred_points_normalized.unsqueeze(0).to(device)

    # Compute metrics
    chamfer_dist = chamfer_distance(pred_points_normalized, gt_points_normalized, w1=1.0, w2=1.0, squared=False).item()
    f_scores = [f_score(pred_points_normalized, gt_points_normalized, radius=t).item() for t in thresholds]

    # Compute voxel IoU
    pred_voxel, gt_voxel, pred_centroid, gt_centroid = normalize_and_align_voxels(pred_path, gt_path, pred_scale_factor, gt_scale_factor)
    voxel_iou = calculate_voxel_iou(gt_voxel, pred_voxel)

    # Plot voxel grids
    voxel_plot_path = os.path.splitext(plot_path)[0] + "_voxels.png"
    plot_voxel_grids(gt_voxel, pred_voxel, pred_centroid, gt_centroid, voxel_plot_path, "Ground Truth vs Predicted Voxel Grids")

    return chamfer_dist, f_scores, voxel_iou

def plot_voxel_grids(voxel_grid1: o3d.geometry.VoxelGrid, voxel_grid2: o3d.geometry.VoxelGrid, 
                     pred_centroid: np.ndarray, gt_centroid: np.ndarray, output_path: str, 
                     title: str = "Voxel Grid Comparison") -> None:
    """
    Visualize two voxel grids side by side and overlaid with IoU analysis.

    Args:
        voxel_grid1 (o3d.geometry.VoxelGrid): Ground truth voxel grid.
        voxel_grid2 (o3d.geometry.VoxelGrid): Predicted voxel grid.
        pred_centroid (np.ndarray): Centroid of the predicted mesh.
        gt_centroid (np.ndarray): Centroid of the ground truth mesh.
        output_path (str): Path to save the plot.
        title (str): Plot title.
    """
    voxels1 = np.array([v.grid_index for v in voxel_grid1.get_voxels()])
    voxels2 = np.array([v.grid_index for v in voxel_grid2.get_voxels()])

    voxel1_centroid = np.mean(voxels1, axis=0) if len(voxels1) > 0 else np.zeros(3)
    voxel2_centroid = np.mean(voxels2, axis=0) if len(voxels2) > 0 else np.zeros(3)
    voxels1_centered = voxels1 - voxel1_centroid
    voxels2_centered = voxels2 - voxel2_centroid

    voxels1_set = set(tuple(v) for v in np.round(voxels1_centered).astype(int))
    voxels2_set = set(tuple(v) for v in np.round(voxels2_centered).astype(int))
    intersection = voxels1_set.intersection(voxels2_set)
    only_in_grid1 = voxels1_set - voxels2_set
    only_in_grid2 = voxels2_set - voxels1_set

    intersection_array = np.array(list(intersection)) if intersection else np.empty((0, 3))
    only_in_grid1_array = np.array(list(only_in_grid1)) if only_in_grid1 else np.empty((0, 3))
    only_in_grid2_array = np.array(list(only_in_grid2)) if only_in_grid2 else np.empty((0, 3))

    all_points = np.vstack([voxels1_centered, voxels2_centered]) if len(voxels1) > 0 and len(voxels2) > 0 else voxels1_centered or voxels2_centered
    max_range = np.max(np.abs(all_points)) * 1.2 if len(all_points) > 0 else 1.0

    fig = plt.figure(figsize=(20, 10))

    # Side-by-side plot
    ax1 = fig.add_subplot(121, projection='3d')
    if len(voxels1_centered) > 0:
        ax1.scatter(voxels1_centered[:, 0], voxels1_centered[:, 1], voxels1_centered[:, 2], 
                    marker='s', s=10, color='blue', label='Ground Truth', alpha=0.5)
    if len(voxels2_centered) > 0:
        ax1.scatter(voxels2_centered[:, 0], voxels2_centered[:, 1], voxels2_centered[:, 2], 
                    marker='s', s=10, color='red', label='Predicted', alpha=0.5)
    ax1.set_title("Ground Truth vs Predicted Voxels")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(-max_range, max_range)
    ax1.set_ylim(-max_range, max_range)
    ax1.set_zlim(-max_range, max_range)
    ax1.legend()
    ax1.set_box_aspect([1, 1, 1])

    # Overlap analysis plot
    ax2 = fig.add_subplot(122, projection='3d')
    if len(intersection_array) > 0:
        ax2.scatter(intersection_array[:, 0], intersection_array[:, 1], intersection_array[:, 2], 
                    marker='s', s=20, color='green', label='Intersection', alpha=0.7)
    if len(only_in_grid1_array) > 0:
        ax2.scatter(only_in_grid1_array[:, 0], only_in_grid1_array[:, 1], only_in_grid1_array[:, 2], 
                    marker='s', s=20, color='blue', label='Only in GT', alpha=0.5)
    if len(only_in_grid2_array) > 0:
        ax2.scatter(only_in_grid2_array[:, 0], only_in_grid2_array[:, 1], only_in_grid2_array[:, 2], 
                    marker='s', s=20, color='red', label='Only in Pred', alpha=0.5)
    
    iou = len(intersection) / len(voxels1_set.union(voxels2_set)) if len(voxels1_set.union(voxels2_set)) > 0 else 0
    ax2.text2D(0.05, 0.95, f'IoU: {iou:.4f}', transform=ax2.transAxes)
    ax2.set_title("Voxel Overlap Analysis")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim(-max_range, max_range)
    ax2.set_ylim(-max_range, max_range)
    ax2.set_zlim(-max_range, max_range)
    ax2.legend()
    ax2.set_box_aspect([1, 1, 1])

    plt.suptitle(title)
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(output_path)
    plt.close()

def plot_point_clouds(points1: torch.Tensor, points2: torch.Tensor, output_path: str, 
                      title: str = "Point Cloud Comparison") -> None:
    """
    Plot two point clouds side by side for comparison.

    Args:
        points1 (torch.Tensor): Ground truth point cloud (N, 3).
        points2 (torch.Tensor): Predicted point cloud (N, 3).
        output_path (str): Path to save the plot.
        title (str): Plot title.
    """
    points1 = points1.cpu().numpy()
    points2 = points2.cpu().numpy()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], c='blue', s=1, alpha=0.6, label='Ground Truth')
    ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], c='red', s=1, alpha=0.6, label='Predicted')
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()

    ax.xaxis._axinfo["grid"]['linewidth'] = 0
    ax.yaxis._axinfo["grid"]['linewidth'] = 0
    ax.zaxis._axinfo["grid"]['linewidth'] = 0

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))



    plt.grid(False)
    plt.savefig(output_path)
    plt.close()

def calculate_averages(metrics_dict: Dict[str, Dict[str, float]]) -> Optional[Dict[str, float]]:
    """
    Calculate the average of each metric across all entries.

    Args:
        metrics_dict (Dict[str, Dict[str, float]]): Dictionary of metrics for each model.

    Returns:
        Optional[Dict[str, float]]: Average metrics or None if the dictionary is empty.
    """
    if not metrics_dict:
        return None

    num_entries = len(metrics_dict)
    avg_metrics = {'chamfer_dist': 0.0, 'f_scores_0.2': 0.0, 'f_scores_0.3': 0.0, 'f_scores_0.5': 0.0, 'volume_IOU': 0.0}

    for metrics in metrics_dict.values():
        avg_metrics['chamfer_dist'] += metrics['chamfer_dist']
        avg_metrics['f_scores_0.2'] += metrics['f_scores_0.2']
        avg_metrics['f_scores_0.3'] += metrics['f_scores_0.3']
        avg_metrics['f_scores_0.5'] += metrics['f_scores_0.5']
        avg_metrics['volume_IOU'] += metrics['volume_IOU']

    for key in avg_metrics:
        avg_metrics[key] /= num_entries

    return avg_metrics

def main() -> None:
    """Main function to evaluate 3D models and compute metrics."""
    parser = argparse.ArgumentParser(description="Evaluate 3D model reconstruction for folders of OBJ files.")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth OBJ file or folder")
    parser.add_argument("--pred", type=str, required=True, help="Path to predicted OBJ file or folder")
    parser.add_argument("--out", type=str, default="evaluation_output", help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device (e.g., cuda:0)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.device.split(":")[-1]))
        print(f"Using GPU: {torch.cuda.current_device()}")
    else:
        print("CUDA is not available. Falling back to CPU.")

    thresholds = [0.2, 0.3, 0.5]
    metrics_dict = {}

    os.makedirs(args.out, exist_ok=True)

    # Check if input is a directory or a single file
    if os.path.isdir(args.gt) and os.path.isdir(args.pred):
        gt_files = sorted([f for f in os.listdir(args.gt) if f.endswith('.obj')])
        pred_files = sorted([f for f in os.listdir(args.pred) if f.endswith('.obj')])
        print(gt_files)
        print(pred_files)
        common_files = sorted(list(set(gt_files) & set(pred_files)))
        if not common_files:
            print("No matching OBJ files found in both folders.")
            return
        print(f"Found {len(common_files)} matching OBJ files.")
        for fname in common_files:
            gt_path = os.path.join(args.gt, fname)
            pred_path = os.path.join(args.pred, fname)
            plot_path = os.path.join(args.out, f"{os.path.splitext(fname)[0]}_comparison.png")
            try:
                chamfer_dist, f_scores, voxel_iou = evaluate_3d_models(
                    gt_path, pred_path, plot_path, thresholds=thresholds, device=args.device
                )
                if chamfer_dist is None or f_scores is None or voxel_iou is None:
                    print(f"Failed to evaluate {fname}.")
                    continue
                metrics = {
                    'chamfer_dist': chamfer_dist,
                    'volume_IOU': voxel_iou,
                    'f_scores_0.2': f_scores[0],
                    'f_scores_0.3': f_scores[1],
                    'f_scores_0.5': f_scores[2]
                }
                metrics_dict[fname] = metrics
                # Save per-mesh metrics
                with open(os.path.join(args.out, f"{os.path.splitext(fname)[0]}_metrics.json"), 'w') as f:
                    json.dump(metrics, f, indent=4)
                print(f"[{fname}] Metrics:\n{json.dumps(metrics, indent=4)}")
            except Exception as e:
                print(f"Error evaluating {fname}: {e}")
        # Save all metrics and averages
        with open(os.path.join(args.out, 'all_metrics.json'), 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        avg_metrics = calculate_averages(metrics_dict)
        if avg_metrics:
            with open(os.path.join(args.out, 'average_metrics.json'), 'w') as f:
                json.dump(avg_metrics, f, indent=4)
            print(f"\nAverage Metrics:\n{json.dumps(avg_metrics, indent=4)}")
        print(f"Evaluation completed. Results saved to {args.out}/")
    else:
        # Single file mode (original behavior)
        plot_path = os.path.join(args.out, "comparison.png")
        try:
            chamfer_dist, f_scores, voxel_iou = evaluate_3d_models(
                args.gt, args.pred, plot_path, thresholds=thresholds, device=args.device
            )
            if chamfer_dist is None or f_scores is None or voxel_iou is None:
                print("Failed to evaluate the models.")
                return

            metrics_dict = {
                'chamfer_dist': chamfer_dist,
                'volume_IOU': voxel_iou,
                'f_scores_0.2': f_scores[0],
                'f_scores_0.3': f_scores[1],
                'f_scores_0.5': f_scores[2]
            }
            print(f"Metrics:\n{json.dumps(metrics_dict, indent=4)}")
        except Exception as e:
            print(f"Error evaluating models: {e}")
            return

        # Save metrics to file
        with open(os.path.join(args.out, 'metrics.json'), 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"Evaluation completed. Metrics saved to {args.out}/metrics.json")
if __name__ == "__main__":
    main()

# python3 evaluation/evaluate.py --gt /home/khoahd/vdt_phase1/data/objaverse_fix1_results --pred /home/khoahd/vdt_phase1/outputs/dreamgaussian_finals --out /home/khoahd/vdt_phase1/outputs/metrics/dreamgaussian