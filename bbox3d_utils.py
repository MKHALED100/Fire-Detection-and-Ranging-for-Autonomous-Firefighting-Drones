import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import KalmanFilter
from collections import defaultdict
import math
from collections import deque  

# Default camera intrinsic matrix (can be overridden)
DEFAULT_K = np.array([
    [718.856, 0.0, 607.1928],
    [0.0, 718.856, 185.2157],
    [0.0, 0.0, 1.0]
])

# Default camera projection matrix (can be overridden)
DEFAULT_P = np.array([
    [718.856, 0.0, 607.1928, 45.38225],
    [0.0, 718.856, 185.2157, -0.1130887],
    [0.0, 0.0, 1.0, 0.003779761]
])

# Average dimensions for common objects (height, width, length) in meters
DEFAULT_DIMS = { 
    'human': np.array([1.75, 0.60, 0.60]),  # Adjusted width/length for person
}

class BBox3DEstimator:
    """
    3D bounding box estimation from 2D detections and depth
    """
    def __init__(self, camera_matrix=None, projection_matrix=None, class_dims=None):
        """
        Initialize the 3D bounding box estimator
        
        Args:
            camera_matrix (numpy.ndarray): Camera intrinsic matrix (3x3)
            projection_matrix (numpy.ndarray): Camera projection matrix (3x4)
            class_dims (dict): Dictionary mapping class names to dimensions (height, width, length)
        """
        self.K = camera_matrix if camera_matrix is not None else DEFAULT_K
        self.P = projection_matrix if projection_matrix is not None else DEFAULT_P
        self.dims = class_dims if class_dims is not None else DEFAULT_DIMS
        
        # Initialize Kalman filters for tracking 3D boxes
        self.kf_trackers = {}
        
        # Store history of 3D boxes for filtering
        self.box_history = defaultdict(list)
        self.max_history = 5
    
    def estimate_3d_box(self, bbox_2d, depth_value, class_name, object_id=None):
        """
        Estimate 3D bounding box from 2D bounding box and depth
        
        Args:
            bbox_2d (list): 2D bounding box [x1, y1, x2, y2]
            depth_value (float): Depth value at the center of the bounding box
            class_name (str): Class name of the object
            object_id (int): Object ID for tracking (None for no tracking)
            
        Returns:
            dict: 3D bounding box parameters
        """
        # Get 2D box center and dimensions
        x1, y1, x2, y2 = bbox_2d
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width_2d = x2 - x1
        height_2d = y2 - y1
        
        # Get dimensions for the class
        if class_name.lower() in self.dims:
            dimensions = self.dims[class_name.lower()].copy()  # Make a copy to avoid modifying the original
        else:
            # Use default human dimensions if class not found
            dimensions = self.dims['human'].copy()
        
        # Adjust dimensions based on 2D box aspect ratio and size
        aspect_ratio_2d = width_2d / height_2d if height_2d > 0 else 1.0
        
        # For plants, adjust dimensions based on 2D box
        if 'plant' in class_name.lower() or 'potted plant' in class_name.lower():
            # Scale height based on 2D box height
            dimensions[0] = height_2d / 120  # Convert pixels to meters with a scaling factor
            # Make width and length proportional to height
            dimensions[1] = dimensions[0] * 0.6  # width
            dimensions[2] = dimensions[0] * 0.6  # length
        
        # For people, adjust dimensions based on 2D box
        elif 'person' in class_name.lower():
            # Scale height based on 2D box height
            dimensions[0] = height_2d / 100  # Convert pixels to meters with a scaling factor
            # Make width and length proportional to height
            dimensions[1] = dimensions[0] * 0.3  # width
            dimensions[2] = dimensions[0] * 0.3  # length
        
        # Convert depth to distance - use a larger range for better visualization
        # Map depth_value (0-1) to a range of 1-10 meters
        distance = 1.0 + depth_value * 9.0  # Increased from 4.0 to 9.0 for a larger range
        
        # Calculate 3D location
        location = self._backproject_point(center_x, center_y, distance)
        
        # For plants, adjust y-coordinate to place them on a surface
        if 'plant' in class_name.lower() or 'potted plant' in class_name.lower():
            # Assume plants are on a surface (e.g., table, floor)
            # Adjust y-coordinate based on the bottom of the 2D bounding box
            bottom_y = y2  # Bottom of the 2D box
            location[1] = self._backproject_point(center_x, bottom_y, distance)[1]
        
        # Estimate orientation
        orientation = self._estimate_orientation(bbox_2d, location, class_name)
        
        # Create 3D box
        box_3d = {
            'dimensions': dimensions,
            'location': location,
            'orientation': orientation,
            'bbox_2d': bbox_2d,
            'object_id': object_id,
            'class_name': class_name
        }
        
        # Apply Kalman filtering if tracking is enabled
        if object_id is not None:
            box_3d = self._apply_kalman_filter(box_3d, object_id)
            
            # Add to history for temporal filtering
            self.box_history[object_id].append(box_3d)
            if len(self.box_history[object_id]) > self.max_history:
                self.box_history[object_id].pop(0)
            
            # Apply temporal filtering
            box_3d = self._apply_temporal_filter(object_id)
        
        return box_3d
    
    def _backproject_point(self, x, y, depth):
        """
        Backproject a 2D point to 3D space
        
        Args:
            x (float): X coordinate in image space
            y (float): Y coordinate in image space
            depth (float): Depth value
            
        Returns:
            numpy.ndarray: 3D point (x, y, z) in camera coordinates
        """
        # Create homogeneous coordinates
        point_2d = np.array([x, y, 1.0])
        
        # Backproject to 3D
        # The z-coordinate is the depth
        # The x and y coordinates are calculated using the inverse of the camera matrix
        point_3d = np.linalg.inv(self.K) @ point_2d * depth
        
        # For indoor scenes, adjust the y-coordinate to be more realistic
        # In camera coordinates, y is typically pointing down
        # Adjust y to place objects at a reasonable height
        # This is a simplification - in a real system, this would be more sophisticated
        point_3d[1] = point_3d[1] * 0.5  # Scale down y-coordinate
        
        return point_3d
    
    def _estimate_orientation(self, bbox_2d, location, class_name):
        """
        Estimate orientation of the object
        
        Args:
            bbox_2d (list): 2D bounding box [x1, y1, x2, y2]
            location (numpy.ndarray): 3D location of the object
            class_name (str): Class name of the object
            
        Returns:
            float: Orientation angle in radians
        """
        # Calculate ray from camera to object center
        theta_ray = np.arctan2(location[0], location[2])
        
        # For plants and stationary objects, orientation doesn't matter much
        # Just use a fixed orientation aligned with the camera view
        if 'plant' in class_name.lower() or 'potted plant' in class_name.lower():
            # Plants typically don't have a specific orientation
            # Just use the ray angle
            return theta_ray
        
        # For people, they might be facing the camera
        if 'person' in class_name.lower():
            # Assume person is facing the camera
            alpha = 0.0
        else:
            # For other objects, use the 2D box aspect ratio to estimate orientation
            x1, y1, x2, y2 = bbox_2d
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 1.0
            
            # If the object is wide, it might be facing sideways
            if aspect_ratio > 1.5:
                # Object is wide, might be facing sideways
                # Use the position relative to the image center to guess orientation
                image_center_x = self.K[0, 2]  # Principal point x
                if (x1 + x2) / 2 < image_center_x:
                    # Object is on the left side of the image
                    alpha = np.pi / 2  # Facing right
                else:
                    # Object is on the right side of the image
                    alpha = -np.pi / 2  # Facing left
            else:
                # Object has normal proportions, assume it's facing the camera
                alpha = 0.0
        
        # Global orientation
        rot_y = alpha + theta_ray
        
        return rot_y
    
    def _init_kalman_filter(self, box_3d):
        """
        Initialize a Kalman filter for a new object
        
        Args:
            box_3d (dict): 3D bounding box parameters
            
        Returns:
            filterpy.kalman.KalmanFilter: Initialized Kalman filter
        """
        # State: [x, y, z, width, height, length, yaw, vx, vy, vz, vyaw]
        kf = KalmanFilter(dim_x=11, dim_z=7)
        
        # Initial state
        kf.x = np.array([
            box_3d['location'][0],
            box_3d['location'][1],
            box_3d['location'][2],
            box_3d['dimensions'][1],  # width
            box_3d['dimensions'][0],  # height
            box_3d['dimensions'][2],  # length
            box_3d['orientation'],
            0, 0, 0, 0  # Initial velocities
        ])
        
        # State transition matrix (motion model)
        dt = 1.0  # Time step
        kf.F = np.eye(11)
        kf.F[0, 7] = dt  # x += vx * dt
        kf.F[1, 8] = dt  # y += vy * dt
        kf.F[2, 9] = dt  # z += vz * dt
        kf.F[6, 10] = dt  # yaw += vyaw * dt
        
        # Measurement function
        kf.H = np.zeros((7, 11))
        kf.H[0, 0] = 1  # x
        kf.H[1, 1] = 1  # y
        kf.H[2, 2] = 1  # z
        kf.H[3, 3] = 1  # width
        kf.H[4, 4] = 1  # height
        kf.H[5, 5] = 1  # length
        kf.H[6, 6] = 1  # yaw
        
        # Measurement uncertainty
        kf.R = np.eye(7) * 0.1
        kf.R[0:3, 0:3] *= 1.0  # Location uncertainty
        kf.R[3:6, 3:6] *= 0.1  # Dimension uncertainty
        kf.R[6, 6] = 0.3  # Orientation uncertainty
        
        # Process uncertainty
        kf.Q = np.eye(11) * 0.1
        kf.Q[7:11, 7:11] *= 0.5  # Velocity uncertainty
        
        # Initial state uncertainty
        kf.P = np.eye(11) * 1.0
        kf.P[7:11, 7:11] *= 10.0  # Velocity uncertainty
        
        return kf
    
    def _apply_kalman_filter(self, box_3d, object_id):
        """
        Apply Kalman filtering to smooth 3D box parameters
        
        Args:
            box_3d (dict): 3D bounding box parameters
            object_id (int): Object ID for tracking
            
        Returns:
            dict: Filtered 3D bounding box parameters
        """
        # Initialize Kalman filter if this is a new object
        if object_id not in self.kf_trackers:
            self.kf_trackers[object_id] = self._init_kalman_filter(box_3d)
        
        # Get the Kalman filter for this object
        kf = self.kf_trackers[object_id]
        
        # Predict
        kf.predict()
        
        # Update with measurement
        measurement = np.array([
            box_3d['location'][0],
            box_3d['location'][1],
            box_3d['location'][2],
            box_3d['dimensions'][1],  # width
            box_3d['dimensions'][0],  # height
            box_3d['dimensions'][2],  # length
            box_3d['orientation']
        ])
        
        kf.update(measurement)
        
        # Update box_3d with filtered values
        filtered_box = box_3d.copy()
        filtered_box['location'] = np.array([kf.x[0], kf.x[1], kf.x[2]])
        filtered_box['dimensions'] = np.array([kf.x[4], kf.x[3], kf.x[5]])  # height, width, length
        filtered_box['orientation'] = kf.x[6]
        
        return filtered_box
    
    def _apply_temporal_filter(self, object_id):
        """
        Apply temporal filtering to smooth 3D box parameters over time
        
        Args:
            object_id (int): Object ID for tracking
            
        Returns:
            dict: Temporally filtered 3D bounding box parameters
        """
        history = self.box_history[object_id]
        
        if len(history) < 2:
            return history[-1]
        
        # Get the most recent box
        current_box = history[-1]
        
        # Apply exponential moving average to location and orientation
        alpha = 0.7  # Weight for current measurement (higher = less smoothing)
        
        # Initialize with current values
        filtered_box = current_box.copy()
        
        # Apply EMA to location and orientation
        for i in range(len(history) - 2, -1, -1):
            weight = alpha * (1 - alpha) ** (len(history) - i - 2)
            filtered_box['location'] = filtered_box['location'] * (1 - weight) + history[i]['location'] * weight
            
            # Handle orientation wrapping
            angle_diff = history[i]['orientation'] - filtered_box['orientation']
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            filtered_box['orientation'] += angle_diff * weight
        
        return filtered_box
    
    def project_box_3d_to_2d(self, box_3d):
        """
        Project 3D bounding box corners to 2D image space
        
        Args:
            box_3d (dict): 3D bounding box parameters
            
        Returns:
            numpy.ndarray: 2D points of the 3D box corners (8x2)
        """
        # Extract parameters
        h, w, l = box_3d['dimensions']
        x, y, z = box_3d['location']
        rot_y = box_3d['orientation']
        class_name = box_3d['class_name'].lower()
        
        # Get 2D box for reference
        x1, y1, x2, y2 = box_3d['bbox_2d']
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width_2d = x2 - x1
        height_2d = y2 - y1
        
        # Create rotation matrix
        R_mat = np.array([
            [np.cos(rot_y), 0, np.sin(rot_y)],
            [0, 1, 0],
            [-np.sin(rot_y), 0, np.cos(rot_y)]
        ])
        
        # 3D bounding box corners
        # For plants and stationary objects, make the box more centered
        if 'plant' in class_name or 'potted plant' in class_name:
            # For plants, center the box on the plant
            x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
            y_corners = np.array([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2])  # Center vertically
            z_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
        else:
            # For other objects, use standard box configuration
            x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
            y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])  # Bottom at y=0
            z_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
        
        # Rotate and translate corners
        corners_3d = np.vstack([x_corners, y_corners, z_corners])
        corners_3d = R_mat @ corners_3d
        corners_3d[0, :] += x
        corners_3d[1, :] += y
        corners_3d[2, :] += z
        
        # Project to 2D
        corners_3d_homo = np.vstack([corners_3d, np.ones((1, 8))])
        corners_2d_homo = self.P @ corners_3d_homo
        corners_2d = corners_2d_homo[:2, :] / corners_2d_homo[2, :]
        
        # Constrain the 3D box to be within a reasonable distance of the 2D box
        # This helps prevent wildly incorrect projections
        mean_x = np.mean(corners_2d[0, :])
        mean_y = np.mean(corners_2d[1, :])
        
        # If the projected box is too far from the 2D box center, adjust it
        if abs(mean_x - center_x) > width_2d or abs(mean_y - center_y) > height_2d:
            # Shift the projected points to center on the 2D box
            shift_x = center_x - mean_x
            shift_y = center_y - mean_y
            corners_2d[0, :] += shift_x
            corners_2d[1, :] += shift_y
        
        return corners_2d.T
    
    def check_intersection(self, x1, y1, x2, y2, person_boxes):
        """Check if fire bounding box intersects with any person bounding box"""
        for px1, py1, px2, py2 in person_boxes:
            if not (x2 < px1 or x1 > px2 or y2 < py1 or y1 > py2):
                return True  # Intersection found
        return False  # No intersection

    



    # def find_fire_target_from_largest_blob(self, frame, x1, x2, y1, y2):
    #     """
    #     Find the center of the bottom region of the largest fire blob.
    #     Fills internal holes, adapts to arc shapes, and refines HSV fire mask.
    #     """
    #     h, w = frame.shape[:2]
    #     x1, y1 = max(0, x1), max(0, y1)
    #     x2, y2 = min(w, x2), min(h, y2)

    #     roi = frame[y1:y2, x1:x2]
    #     hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    #     # Updated HSV range for better yellow + orange fire capture
    #     lower_fire = np.array([8, 60, 160])
    #     upper_fire = np.array([55, 255, 255])
    #     mask = cv2.inRange(hsv, lower_fire, upper_fire)

    #     # Morphological cleanup
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #     mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #     mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

    #     # Find valid contours
    #     contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]

    #     if not contours:
    #         return (x1 + x2) // 2, y2 - (y2 - y1) // 8

    #     largest = max(contours, key=cv2.contourArea)

    #     # Fill contour to get solid blob without internal holes
    #     fire_blob_mask = np.zeros_like(mask)
    #     cv2.drawContours(fire_blob_mask, [largest], -1, 255, -1)

    #     # Fill any small internal holes in the blob
    #     fire_blob_mask = cv2.morphologyEx(fire_blob_mask, cv2.MORPH_CLOSE, kernel)

    #     # Focus on bottom quarter
    #     x, y, w_blob, h_blob = cv2.boundingRect(largest)
    #     quarter_start_y = y + int(0.75 * h_blob)
    #     bottom_mask = np.zeros_like(fire_blob_mask)
    #     bottom_mask[quarter_start_y:y + h_blob, :] = fire_blob_mask[quarter_start_y:y + h_blob, :]

    #     ys, xs = np.where(bottom_mask > 0)
    #     fallback_cx = x + w_blob // 2
    #     fallback_cy = y + h_blob - (h_blob // 8)

    #     # If fallback center is valid
    #     if (0 <= fallback_cx < roi.shape[1]) and (0 <= fallback_cy < roi.shape[0]) and \
    #     fire_blob_mask[fallback_cy, fallback_cx] > 0:
    #         cx, cy = fallback_cx, fallback_cy
    #     elif len(xs) > 0:
    #         fire_pixels = np.column_stack((xs, ys))
    #         dists = np.linalg.norm(fire_pixels - np.array([fallback_cx, fallback_cy]), axis=1)
    #         cx, cy = fire_pixels[np.argmin(dists)]
    #     else:
    #         cx, cy = fallback_cx, fallback_cy

    #     fire_x = x1 + cx
    #     fire_y = y1 + cy

    #     cv2.imshow("Fire Mask", fire_blob_mask)
    #     cv2.waitKey(1)
    #     return fire_x, fire_y


    def find_fire_target_from_largest_blob(self, frame, x1, x2, y1, y2):
        """
        Improved fire source estimation:
        - Uses bottom region of largest fire blob
        - Weights toward lower pixels
        - Favors yellow fire pixels (hottest zone)
        """
        import cv2
        import numpy as np

        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Fire range: broader orange to yellow
        lower_fire = np.array([8, 60, 160])
        upper_fire = np.array([55, 255, 255])
        mask = cv2.inRange(hsv, lower_fire, upper_fire)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]

        if not contours:
            return (x1 + x2) // 2, y2 - (y2 - y1) // 8

        largest = max(contours, key=cv2.contourArea)

        # Fill the largest fire blob
        fire_blob_mask = np.zeros_like(mask)
        cv2.drawContours(fire_blob_mask, [largest], -1, 255, -1)
        fire_blob_mask = cv2.morphologyEx(fire_blob_mask, cv2.MORPH_CLOSE, kernel)

        # Focus on bottom quarter
        x, y, w_blob, h_blob = cv2.boundingRect(largest)
        quarter_start_y = y + int(0.75 * h_blob)
        bottom_mask = np.zeros_like(fire_blob_mask)
        bottom_mask[quarter_start_y:y + h_blob, :] = fire_blob_mask[quarter_start_y:y + h_blob, :]

        # Weighted centroid: favor yellow fire
        ys, xs = np.where(bottom_mask > 0)
        fallback_cx = x + w_blob // 2
        fallback_cy = y + h_blob - (h_blob // 8)

        if len(xs) > 0:
            # Convert mask coordinates to HSV pixel indices
            hsv_mask = hsv[:, :, :]  # same shape as ROI

            # Build weight based on color temperature: favor yellow (H ~ 25-35)
            hs = hsv[ys, xs, 0]  # hue values at fire pixels
            vs = hsv[ys, xs, 2]  # brightness values
            # High weight for hue between 25-35 (yellow), else downscale
            hue_weights = np.exp(-((hs - 30) ** 2) / (2 * 6**2))  # Gaussian favoring ~30
            vertical_weights = np.linspace(0, 1, fire_blob_mask.shape[0])[ys]  # bottom-heavy

            combined_weights = hue_weights * vertical_weights * (vs / 255.0)

            try:
                cx = int(np.average(xs, weights=combined_weights))
                cy = int(np.average(ys, weights=combined_weights))
            except ZeroDivisionError:
                cx, cy = fallback_cx, fallback_cy
        else:
            cx, cy = fallback_cx, fallback_cy

        fire_x = x1 + cx
        fire_y = y1 + cy

        # Debug output
        debug = cv2.cvtColor(fire_blob_mask, cv2.COLOR_GRAY2BGR)
        cv2.circle(debug, (cx, cy), 5, (0, 0, 255), -1)
        cv2.imshow("Fire Mask with Yellow-Weighted Centroid", debug)
        cv2.waitKey(1)

        return fire_x, fire_y





    def find_fire_core_bottom_brightest(self, frame, x1, x2, y1, y2):
        """
        Locate the brightest fire-like pixel in the lower quarter of the bounding box.
        """
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Fire-like colors in HSV (tweakable)
        lower_fire = np.array([5, 100, 100])
        upper_fire = np.array([35, 255, 255])

        mask = cv2.inRange(hsv, lower_fire, upper_fire)

        # Focus only on the lower quarter of the mask
        h_roi = mask.shape[0]
        y_start = int(h_roi * 0.75)  # bottom 25%
        bottom_mask = np.zeros_like(mask)
        bottom_mask[y_start:] = mask[y_start:]

        # Apply the bottom mask to the grayscale version
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        fire_gray = cv2.bitwise_and(gray, gray, mask=bottom_mask)

        # Find the brightest pixel in this bottom-fire region
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(fire_gray)

        if max_val == 0:
            # Fallback: center-bottom of the box
            return (x1 + x2) // 2, y2 - (y2 - y1) // 8
        else:
            fire_x = x1 + max_loc[0]
            fire_y = y1 + max_loc[1]
            return fire_x, fire_y


    def draw_target_marker_at(self, frame, x, y, color):
        """Draw a sniper-style target marker at a specific (x, y) location."""
        radius = 10
        line_len = 6

        # Draw circle
        cv2.circle(frame, (x, y), radius, color, 2)

        # Draw crosshairs
        cv2.line(frame, (x - line_len, y), (x + line_len, y), color, 2)
        cv2.line(frame, (x, y - line_len), (x, y + line_len), color, 2)


    
    def draw_box_3d(self, image, box_3d, color=(0, 255, 0), thickness=2):
        """
        Draw enhanced 3D bounding box on image with better depth perception
        
        Args:
            image (numpy.ndarray): Image to draw on
            box_3d (dict): 3D bounding box parameters
            color (tuple): Color in BGR format
            thickness (int): Line thickness
            
        Returns:
            numpy.ndarray: Image with 3D box drawn
        """
        # Get 2D box coordinates
        x1, y1, x2, y2 = [int(coord) for coord in box_3d['bbox_2d']]
        
        # Get depth value for scaling
        depth_value = box_3d.get('depth_value', 0.5)
        
        # Calculate box dimensions
        width = x2 - x1
        height = y2 - y1
        
        # Calculate the offset for the 3D effect (deeper objects have smaller offset)
        # Inverse relationship with depth - closer objects have larger offset
        offset_factor = 1.0 - depth_value
        offset_x = int(width * 0.3 * offset_factor)
        offset_y = int(height * 0.3 * offset_factor)
        
        # Ensure minimum offset for visibility
        offset_x = max(8, min(offset_x, 50))
        offset_y = max(8, min(offset_y, 50))
        
        # Create points for the 3D box
        # Front face (the 2D bounding box)
        front_tl = (x1, y1)
        front_tr = (x2, y1)
        front_br = (x2, y2)
        front_bl = (x1, y2)
        
        # Back face (offset by depth)
        back_tl = (x1 + offset_x, y1 - offset_y)
        back_tr = (x2 + offset_x, y1 - offset_y)
        back_br = (x2 + offset_x, y2 - offset_y)
        back_bl = (x1 + offset_x, y2 - offset_y)
        
        # Create a slightly transparent copy of the image for the 3D effect
        overlay = image.copy()
        
        # Draw the front face (2D bounding box)
        cv2.rectangle(image, front_tl, front_br, color, thickness)
        
        # # Draw the connecting lines between front and back faces
        # cv2.line(image, front_tl, back_tl, color, thickness)
        # cv2.line(image, front_tr, back_tr, color, thickness)
        # cv2.line(image, front_br, back_br, color, thickness)
        # cv2.line(image, front_bl, back_bl, color, thickness)
        
        # # Draw the back face
        # cv2.line(image, back_tl, back_tr, color, thickness)
        # cv2.line(image, back_tr, back_br, color, thickness)
        # cv2.line(image, back_br, back_bl, color, thickness)
        # cv2.line(image, back_bl, back_tl, color, thickness)
        
        # Fill the top face with a semi-transparent color to enhance 3D effect
        pts_top = np.array([front_tl, front_tr, back_tr, back_tl], np.int32)
        pts_top = pts_top.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts_top], color)
        
        # Fill the right face with a semi-transparent color
        pts_right = np.array([front_tr, front_br, back_br, back_tr], np.int32)
        pts_right = pts_right.reshape((-1, 1, 2))
        # Darken the right face color for better 3D effect
        right_color = (int(color[0] * 0.7), int(color[1] * 0.7), int(color[2] * 0.7))
        cv2.fillPoly(overlay, [pts_right], right_color)
        
        # Apply the overlay with transparency
        alpha = 0.3  # Transparency factor
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # Get class name and object ID
        class_name = box_3d['class_name']
        obj_id = box_3d['object_id'] if 'object_id' in box_3d else None
        
        # Draw text information
        text_y = y1 - 10
        if obj_id is not None:
            cv2.putText(image, f"ID:{obj_id}", (x1, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            text_y -= 15
        
        cv2.putText(image, class_name, (x1, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        text_y -= 15
        
        # Get depth information if available
        if 'depth_value' in box_3d:
            depth_value = box_3d['depth_value']
            depth_method = box_3d.get('depth_method', 'unknown')
            depth_text = f"D:{depth_value:.2f} ({depth_method})"
            cv2.putText(image, depth_text, (x1, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            text_y -= 15
        
        # Get score if available
        if 'score' in box_3d:
            score = box_3d['score']
            score_text = f"S:{score:.2f}"
            cv2.putText(image, score_text, (x1, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw a vertical line from the bottom of the box to the ground
        # This helps with depth perception
        ground_y = y2 + int(height * 0.2)  # A bit below the bottom of the box
        cv2.line(image, (int((x1 + x2) / 2), y2), (int((x1 + x2) / 2), ground_y), color, thickness)
        
        # Draw a small circle at the bottom to represent the ground contact point
        cv2.circle(image, (int((x1 + x2) / 2), ground_y), thickness * 2, color, -1)
        
        # # Draw target marker if this is a fire object
        # if 'fire' in class_name.lower():
        #     fire_x, fire_y = self.find_fire_target_from_largest_blob(image, x1, x2, y1, y2)
        #     self.draw_target_marker_at(image, fire_x, fire_y, color)



        
        return image
    
 

    def draw_tracking_trajectories(self, frame, tracking_trajectories):
        """
        Draw tracking trajectories on the result frame.

        Args:
            frame (numpy.ndarray): The frame to draw on.
            tracking_trajectories (dict): Object ID to list of (x, y) centroid tuples.

        Returns:
            numpy.ndarray: Frame with trajectories drawn.
        """
        for id_, trajectory in tracking_trajectories.items():
            for i in range(1, len(trajectory)):
                thickness = int(2 * (i / len(trajectory)) + 1)
                cv2.line(frame,
                        (int(trajectory[i - 1][0]), int(trajectory[i - 1][1])),
                        (int(trajectory[i][0]), int(trajectory[i][1])),
                        (255, 255, 255), thickness)  
        return frame


    
    def cleanup_trackers(self, active_ids):
        """
        Clean up Kalman filters and history for objects that are no longer tracked
        
        Args:
            active_ids (list): List of active object IDs
        """
        # Convert to set for faster lookup
        active_ids_set = set(active_ids)
        
        # Clean up Kalman filters
        for obj_id in list(self.kf_trackers.keys()):
            if obj_id not in active_ids_set:
                del self.kf_trackers[obj_id]
        
        # Clean up box history
        for obj_id in list(self.box_history.keys()):
            if obj_id not in active_ids_set:
                del self.box_history[obj_id]

class BirdEyeView:
    """
    Bird's Eye View visualization
    """
    def __init__(self, size=(400, 400), scale=30, camera_height=1.2, camera_image_size=(1920, 1080)):
        """
        Initialize the Bird's Eye View visualizer
        
        Args:
            size (tuple): Size of the BEV image (width, height)
            scale (float): Scale factor (pixels per meter)
            camera_height (float): Height of the camera above ground (meters)
        """
        self.width, self.height = size
        self.scale = scale
        self.camera_height = camera_height
        self.camera_width, self.camera_height_input = camera_image_size
        
        # Create empty BEV image
        self.bev_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Set origin at the bottom center of the image
        self.origin_x = self.width // 2
        self.origin_y = self.height - 50
    
    def reset(self):
        """
        Reset the BEV image
        """
        # Create a dark background
        self.bev_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.bev_image[:, :] = (20, 20, 20)  # Dark gray background
        
        # Draw grid lines
        grid_spacing = max(int(self.scale), 20)  # At least 20 pixels between grid lines
        
        # Draw horizontal grid lines
        for y in range(self.origin_y, 0, -grid_spacing):
            cv2.line(self.bev_image, (0, y), (self.width, y), (50, 50, 50), 1)
        
        # Draw vertical grid lines
        for x in range(0, self.width, grid_spacing):
            cv2.line(self.bev_image, (x, 0), (x, self.height), (50, 50, 50), 1)
        
        # Draw coordinate system
        axis_length = min(80, self.height // 5)
        
        # X-axis (upward)
        cv2.line(self.bev_image, 
                (self.origin_x, self.origin_y), 
                (self.origin_x, self.origin_y - axis_length), 
                (0, 200, 0), 2)  # Green for X-axis
        
        # Y-axis (rightward)
        cv2.line(self.bev_image, 
                (self.origin_x, self.origin_y), 
                (self.origin_x + axis_length, self.origin_y), 
                (0, 0, 200), 2)  # Red for Y-axis
        
        # Add axis labels
        cv2.putText(self.bev_image, "X", 
                   (self.origin_x - 15, self.origin_y - axis_length + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        
        cv2.putText(self.bev_image, "Y", 
                   (self.origin_x + axis_length - 15, self.origin_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
        
        # Draw distance markers specifically for 1-5 meter range
        # Use fixed steps of 1 meter with intermediate markers at 0.5 meters
        for dist in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]:
            y = self.origin_y - int(dist * self.scale)
            
            if y < 20:  # Skip if too close to top
                continue
            
            # Draw tick mark - thicker for whole meters
            thickness = 2 if dist.is_integer() else 1
            cv2.line(self.bev_image, 
                    (self.origin_x - 5, y), 
                    (self.origin_x + 5, y), 
                    (120, 120, 120), thickness)
            
            # Only show text for whole meters
            if dist.is_integer():
                cv2.putText(self.bev_image, f"{int(dist)}m", 
                           (self.origin_x + 10, y + 4), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    
    def draw_box(self, box_3d, color=None):
        """
        Draw a more realistic representation of an object on the BEV image
        
        Args:
            box_3d (dict): 3D bounding box parameters
            color (tuple): Color in BGR format (None for automatic color based on class)
        """
        try:
            # Extract parameters
            class_name = box_3d['class_name'].lower()
            
            # Scale depth to fit within 1-5 meters range
            depth_value = box_3d.get('depth_value', 0.5)
            # print(depth_value)
            # Map depth value (0-1) to a range of 1-5 meters
            depth = depth_value
            
            # Get 2D box dimensions for size estimation
            if 'bbox_2d' in box_3d:
                x1, y1, x2, y2 = box_3d['bbox_2d']
                width_2d = x2 - x1
                height_2d = y2 - y1
                size_factor = width_2d / 100
                size_factor = max(0.5, min(size_factor, 2.0))
            else:
                size_factor = 1.0
            
            # Determine color based on class
            if color is None:
                if 'human' in class_name:
                    color = (255, 0, 0) # Green
                elif 'fire' :
                    # Check if fire is intersecting
                    intersecting = box_3d.get("intersecting", False)
                    color = (0, 0, 255) if intersecting else (0, 255, 0)  # Red if intersecting, green if safe
                else:
                    color = (255, 255, 255)  # White
            
            # Get object ID if available
            obj_id = box_3d.get('object_id', None)
            
            # Calculate position in BEV with flipped axes
            # X-axis points upward, Y-axis points rightward
            
            # Calculate Y position (upward) based on depth
            bev_y = self.origin_y - int(depth * self.scale)
            
            # Calculate X position relative to center of the image
            if 'bbox_2d' in box_3d:
                center_x_2d = (x1 + x2) / 2
                image_width = self.bev_image.shape[1]
                rel_x = center_x_2d / self.camera_width  # -0.5 (left), 0 (center), +0.5 (right)
                bev_x = int(rel_x * self.width)
                
            else:
                bev_x = 0

            
            # Ensure the object stays within the visible area
            bev_x = max(20, min(bev_x, self.width - 20))
            bev_y = max(20, min(bev_y, self.origin_y - 10))
            # print(f"y coordinate: {bev_y}")
            
            # Draw object based on type
            if 'human' in class_name:
                # Draw person as a circle
                radius = int(4 * size_factor)
                cv2.circle(self.bev_image, (bev_x, bev_y), radius, color, -1)
                
            elif 'fire' in class_name:
                # Draw fire as a red triangle           
                triangle_height = int(36 * size_factor)
                triangle_base = int(24 * size_factor)

                # Define triangle vertices (pointing upward)
                pt1 = (bev_x, bev_y - triangle_height // 2)  # top point
                pt2 = (bev_x - triangle_base // 2, bev_y + triangle_height // 2)  # bottom left
                pt3 = (bev_x + triangle_base // 2, bev_y + triangle_height // 2)  # bottom right
                triangle_cnt = np.array([pt1, pt2, pt3])

                cv2.fillPoly(self.bev_image, [triangle_cnt], color)  # red color
                
                
            else:
                # Default: draw a square for other objects
                size = int(8 * size_factor)
                cv2.rectangle(self.bev_image,
                             (bev_x - size, bev_y - size),
                             (bev_x + size, bev_y + size),
                             (255, 255, 255), -1)
            
            # Draw object ID if available
            if obj_id is not None:
                cv2.putText(self.bev_image, f"{obj_id}", 
                           (bev_x - 5, bev_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw distance line from origin to object
            cv2.line(self.bev_image, 
                    (self.origin_x, self.origin_y),
                    (bev_x, bev_y),
                    (70, 70, 70), 1)
        except Exception as e:
            print(f"Error drawing box in BEV: {e}")
    
    def get_image(self):
        """
        Get the BEV image
        
        Returns:
            numpy.ndarray: BEV image
        """
        return self.bev_image 