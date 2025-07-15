import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from transformers import pipeline
from PIL import Image
from zoedepth.utils.misc import colorize as zoe_colorize

# Add ZoeDepth to path if needed
import sys
sys.path.insert(0, "ZoeDepth")
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

class DepthEstimator:
    def __init__(self, model_size='small', device=None, backend='depth-anything'):
        """
        Initialize the depth estimator

        Args:
            model_size (str): Model size (ZoeDepth: 'infer' or 'eval', Depth Anything: 'small', 'base', 'large')
            device (str): Device to run inference on ('cuda', 'cpu', 'mps')
            backend (str): 'depth-anything' or 'zoedepth'
        """
        self.backend = backend.lower()

        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = device

        if self.backend == 'zoedepth':
            # Load ZoeDepth
            self.conf = get_config("zoedepth_nk", model_size)
            self.model = build_model(self.conf).to(self.device).eval()
            print("Loaded ZoeDepth model")
        elif self.backend == 'depth-anything':
            # Load Depth Anything
            model_map = {
                'small': 'depth-anything/Depth-Anything-V2-Small-hf',
                'base': 'depth-anything/Depth-Anything-V2-Base-hf',
                'large': 'depth-anything/Depth-Anything-V2-Large-hf'
            }
            model_name = model_map.get(model_size.lower(), model_map['small'])
            try:
                self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.device)
                print(f"Loaded Depth Anything v2 {model_size} model")
            except Exception as e:
                print(f"Failed loading Depth Anything on {self.device}, fallback to CPU: {e}")
                self.pipe = pipeline(task="depth-estimation", model=model_name, device='cpu')
        else:
            raise ValueError("Unsupported backend. Use 'zoedepth' or 'depth-anything'")
        
    def laser_scale(self, x, y):
        """
        Placeholder for laser calibration. Will later be replaced by actual laser reading.
        """
        return 1.5  # meters for now


    def estimate_depth(self, image):
        """
        Estimate depth from an image
        
        Args:
            image (numpy.ndarray): Input image (BGR format)
            
        Returns:
            numpy.ndarray: Depth map (normalized to 0-1)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)

        if self.backend == 'zoedepth':
            with torch.no_grad():
                depth = self.model.infer_pil(pil_image)  # Real-world meters
            return depth

        elif self.backend == 'depth-anything':
            result = self.pipe(pil_image)
            depth = result['depth']
            if isinstance(depth, Image.Image):
                depth = np.array(depth)
            elif isinstance(depth, torch.Tensor):
                depth = depth.cpu().numpy()

            # Normalize to [0, 1]
            depth_min = depth.min()
            depth_max = depth.max()
            if depth_max > depth_min:
                depth = (depth - depth_min) / (depth_max - depth_min)

            # Invert so closer = smaller values
            depth = 1.0 - depth

            # Find a representative depth value from the closest region
            flat_depth = depth.flatten()
            sorted_indices = np.argsort(flat_depth)
            top_percent = int(0.01 * len(flat_depth))  # Top 1% closest points
            closest_vals = flat_depth[sorted_indices[:top_percent]]
            depth_ref = np.median(closest_vals)

            # Find a pixel near that median depth value
            coords = np.argwhere(np.isclose(depth, depth_ref, atol=1e-3))
            if len(coords) > 0:
                ref_y, ref_x = coords[0]
            else:
                ref_y, ref_x = depth.shape[0] // 2, depth.shape[1] // 2  # fallback

            # Get scale from laser
            real_distance = self.laser_scale(ref_x, ref_y)


            # Scale depth map to approximate metric depth
            if depth_ref > 0:
                scale = 3.0        #real_distance / depth_ref
                # print(f"scale:{scale}")
                # print(f"depth before scaling {depth[ref_y, ref_x]}")
                depth = depth * scale
                # print(f"depth after scaling: {depth[ref_y, ref_x]}")

            return depth*1.79

        
    def colorize_depth(self, depth_map, cmap=cv2.COLORMAP_INFERNO):
        """
        Colorize depth map for visualization
        
        Args:
            depth_map (numpy.ndarray): Depth map (normalized to 0-1)
            cmap (int): OpenCV colormap
            
        Returns:
            numpy.ndarray: Colorized depth map (BGR format)
        """
        
        # Normalize a copy for visualization
        depth_vis = depth_map.copy()
        depth_min = np.min(depth_vis)
        depth_max = np.max(depth_vis)
        if depth_max > depth_min:
            depth_vis = (depth_vis - depth_min) / (depth_max - depth_min)
        depth_map_uint8 = (depth_vis * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_map_uint8, cmap)


    def get_depth_at_point(self, depth_map, x, y):
        """
        Get depth value at a specific point
        
        Args:
            depth_map (numpy.ndarray): Depth map
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            float: Depth value at (x, y)
        """
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            return depth_map[y, x]
        return 0.0
    
    def get_depth_in_region(self, depth_map, bbox, method='median'):
        """
        Get depth value in a region defined by a bounding box
        
        Args:
            depth_map (numpy.ndarray): Depth map
            bbox (list): Bounding box [x1, y1, x2, y2]
            method (str): Method to compute depth ('median', 'mean', 'min')
            
        Returns:
            float: Depth value in the region
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_map.shape[1] - 1, x2)
        y2 = min(depth_map.shape[0] - 1, y2)
        
        # Extract region
        region = depth_map[y1:y2, x1:x2]
        
        if region.size == 0:
            return 0.0
        
        # Compute depth based on method
        if method == 'median':
            return float(np.median(region))
        elif method == 'mean':
            return float(np.mean(region))
        elif method == 'min':
            return float(np.min(region))
        else:
            return float(np.median(region)) 