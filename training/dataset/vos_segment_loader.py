# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import os

import numpy as np
import pandas as pd
import torch

from PIL import Image as PILImage

try:
    from pycocotools import mask as mask_utils
except:
    pass


class JSONSegmentLoader:
    def __init__(self, video_json_path, ann_every=1, frames_fps=24, valid_obj_ids=None):
        # Annotations in the json are provided every ann_every th frame
        self.ann_every = ann_every
        # Ids of the objects to consider when sampling this video
        self.valid_obj_ids = valid_obj_ids
        with open(video_json_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                self.frame_annots = data
            elif isinstance(data, dict):
                masklet_field_name = "masklet" if "masklet" in data else "masks"
                self.frame_annots = data[masklet_field_name]
                if "fps" in data:
                    if isinstance(data["fps"], list):
                        annotations_fps = int(data["fps"][0])
                    else:
                        annotations_fps = int(data["fps"])
                    assert frames_fps % annotations_fps == 0
                    self.ann_every = frames_fps // annotations_fps
            else:
                raise NotImplementedError

    def load(self, frame_id, obj_ids=None):
        assert frame_id % self.ann_every == 0
        rle_mask = self.frame_annots[frame_id // self.ann_every]

        valid_objs_ids = set(range(len(rle_mask)))
        if self.valid_obj_ids is not None:
            # Remove the masklets that have been filtered out for this video
            valid_objs_ids &= set(self.valid_obj_ids)
        if obj_ids is not None:
            # Only keep the objects that have been sampled
            valid_objs_ids &= set(obj_ids)
        valid_objs_ids = sorted(list(valid_objs_ids))

        # Construct rle_masks_filtered that only contains the rle masks we are interested in
        id_2_idx = {}
        rle_mask_filtered = []
        for obj_id in valid_objs_ids:
            if rle_mask[obj_id] is not None:
                id_2_idx[obj_id] = len(rle_mask_filtered)
                rle_mask_filtered.append(rle_mask[obj_id])
            else:
                id_2_idx[obj_id] = None

        # Decode the masks
        raw_segments = torch.from_numpy(mask_utils.decode(rle_mask_filtered)).permute(
            2, 0, 1
        )  # （num_obj, h, w）
        segments = {}
        for obj_id in valid_objs_ids:
            if id_2_idx[obj_id] is None:
                segments[obj_id] = None
            else:
                idx = id_2_idx[obj_id]
                segments[obj_id] = raw_segments[idx]
        return segments

    def get_valid_obj_frames_ids(self, num_frames_min=None):
        # For each object, find all the frames with a valid (not None) mask
        num_objects = len(self.frame_annots[0])

        # The result dict associates each obj_id with the id of its valid frames
        res = {obj_id: [] for obj_id in range(num_objects)}

        for annot_idx, annot in enumerate(self.frame_annots):
            for obj_id in range(num_objects):
                if annot[obj_id] is not None:
                    res[obj_id].append(int(annot_idx * self.ann_every))

        if num_frames_min is not None:
            # Remove masklets that have less than num_frames_min valid masks
            for obj_id, valid_frames in list(res.items()):
                if len(valid_frames) < num_frames_min:
                    res.pop(obj_id)

        return res


class PalettisedPNGSegmentLoader:
    def __init__(self, video_png_root):
        """
        SegmentLoader for datasets with masks stored as palettised PNGs.
        video_png_root: the folder contains all the masks stored in png
        """
        self.video_png_root = video_png_root
        # build a mapping from frame id to their PNG mask path
        # note that in some datasets, the PNG paths could have more
        # than 5 digits, e.g. "00000000.png" instead of "00000.png"
        png_filenames = os.listdir(self.video_png_root)
        self.frame_id_to_png_filename = {}
        for filename in png_filenames:
            frame_id, _ = os.path.splitext(filename)
            self.frame_id_to_png_filename[int(frame_id)] = filename

    def load(self, frame_id):
        """
        load the single palettised mask from the disk (path: f'{self.video_png_root}/{frame_id:05d}.png')
        Args:
            frame_id: int, define the mask path
        Return:
            binary_segments: dict
        """
        # check the path
        mask_path = os.path.join(
            self.video_png_root, self.frame_id_to_png_filename[frame_id]
        )

        # load the mask
        masks = PILImage.open(mask_path).convert("RGB")
        masks = np.array(masks)

        object_id = pd.DataFrame(masks.reshape(-1, 3), columns=['R', 'G', 'B']).drop_duplicates().values
        # object_id = pd.unique(masks.flatten())
        # object_id = object_id[object_id != 0]  # remove background (0)

        # convert into N binary segmentation masks
        binary_segments = {}
        for i in object_id:
            # bs = masks == i
            # binary_segments[i] = torch.from_numpy(bs)

            # check if i != [0,0,0]
            if np.all(i == 0):
                continue
            mask = (masks[..., 0] == i[0]) & (masks[..., 1] == i[1]) & (masks[..., 2] == i[2])
            # convert i to a hashable tuple
            r, g, b = tuple(i.tolist())
            obj_id = (r << 16) | (g << 8) | b
            # # Inverse of the encoding
            # r_i = (obj_id >> 16) & 0xFF
            # g_i = (obj_id >> 8) & 0xFF
            # b_i = obj_id & 0xFF
            binary_segments[obj_id] = torch.from_numpy(mask)

        return binary_segments

    def __len__(self):
        return


class MultiplePNGSegmentLoader:
    def __init__(self, video_png_root, single_object_mode=False):
        """
        video_png_root: the folder contains all the masks stored in png
        single_object_mode: whether to load only a single object at a time
        """
        self.video_png_root = video_png_root
        self.single_object_mode = single_object_mode
        # read a mask to know the resolution of the video
        if self.single_object_mode:
            tmp_mask_path = glob.glob(os.path.join(video_png_root, "*.png"))[0]
        else:
            tmp_mask_path = glob.glob(os.path.join(video_png_root, "*", "*.png"))[0]
        tmp_mask = np.array(PILImage.open(tmp_mask_path))
        self.H = tmp_mask.shape[0]
        self.W = tmp_mask.shape[1]
        if self.single_object_mode:
            self.obj_id = (
                int(video_png_root.split("/")[-1]) + 1
            )  # offset by 1 as bg is 0
        else:
            self.obj_id = None

    def load(self, frame_id):
        if self.single_object_mode:
            return self._load_single_png(frame_id)
        else:
            return self._load_multiple_pngs(frame_id)

    def _load_single_png(self, frame_id):
        """
        load single png from the disk (path: f'{self.obj_id}/{frame_id:05d}.png')
        Args:
            frame_id: int, define the mask path
        Return:
            binary_segments: dict
        """
        mask_path = os.path.join(self.video_png_root, f"{frame_id:05d}.png")
        binary_segments = {}

        if os.path.exists(mask_path):
            mask = np.array(PILImage.open(mask_path))
        else:
            # if png doesn't exist, empty mask
            mask = np.zeros((self.H, self.W), dtype=bool)
        binary_segments[self.obj_id] = torch.from_numpy(mask > 0)
        return binary_segments

    def _load_multiple_pngs(self, frame_id):
        """
        load multiple png masks from the disk (path: f'{obj_id}/{frame_id:05d}.png')
        Args:
            frame_id: int, define the mask path
        Return:
            binary_segments: dict
        """
        # get the path
        all_objects = sorted(glob.glob(os.path.join(self.video_png_root, "*")))
        num_objects = len(all_objects)
        assert num_objects > 0

        # load the masks
        binary_segments = {}
        for obj_folder in all_objects:
            # obj_folder is {video_name}/{obj_id}, obj_id is specified by the name of the folder
            obj_id = int(obj_folder.split("/")[-1])
            obj_id = obj_id + 1  # offset 1 as bg is 0
            mask_path = os.path.join(obj_folder, f"{frame_id:05d}.png")
            if os.path.exists(mask_path):
                mask = np.array(PILImage.open(mask_path))
            else:
                mask = np.zeros((self.H, self.W), dtype=bool)
            binary_segments[obj_id] = torch.from_numpy(mask > 0)

        return binary_segments

    def __len__(self):
        return


class LazySegments:
    """
    Only decodes segments that are actually used.
    """

    def __init__(self):
        self.segments = {}
        self.cache = {}

    def __setitem__(self, key, item):
        self.segments[key] = item

    def __getitem__(self, key):
        if key in self.cache:
            return self.cache[key]
        rle = self.segments[key]
        mask = torch.from_numpy(mask_utils.decode([rle])).permute(2, 0, 1)[0]
        self.cache[key] = mask
        return mask

    def __contains__(self, key):
        return key in self.segments

    def __len__(self):
        return len(self.segments)

    def keys(self):
        return self.segments.keys()


class SA1BSegmentLoader:
    def __init__(
        self,
        video_mask_path,
        mask_area_frac_thresh=1.1,
        video_frame_path=None,
        uncertain_iou=-1,
    ):
        with open(video_mask_path, "r") as f:
            self.frame_annots = json.load(f)

        if mask_area_frac_thresh <= 1.0:
            # Lazily read frame
            orig_w, orig_h = PILImage.open(video_frame_path).size
            area = orig_w * orig_h

        self.frame_annots = self.frame_annots["annotations"]

        rle_masks = []
        for frame_annot in self.frame_annots:
            if not frame_annot["area"] > 0:
                continue
            if ("uncertain_iou" in frame_annot) and (
                frame_annot["uncertain_iou"] < uncertain_iou
            ):
                # uncertain_iou is stability score
                continue
            if (
                mask_area_frac_thresh <= 1.0
                and (frame_annot["area"] / area) >= mask_area_frac_thresh
            ):
                continue
            rle_masks.append(frame_annot["segmentation"])

        self.segments = LazySegments()
        for i, rle in enumerate(rle_masks):
            self.segments[i] = rle

    def load(self, frame_idx):
        return self.segments
    

from shapely.geometry import shape, box
from shapely.ops import unary_union
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

class GeoJSONSegmentLoader:
    def __init__(self, video_geojson_root):
        """
        SegmentLoader for datasets with masks stored as GeoJSON format.
        video_geojson_root: the folder contains all the masks stored in GeoJSON
        """
        self.video_geojson_root = video_geojson_root
        
        # Load metadata for transformation matrix
        meta_files = [f for f in os.listdir(self.video_geojson_root) 
                     if f.endswith('_meta.json')]
        if meta_files:
            meta_path = os.path.join(self.video_geojson_root, meta_files[0])
            with open(meta_path, 'r') as f:
                self.transform_matrix = np.array(json.load(f)["dxf2img"])
        else:
            raise ValueError("No metadata file found in the video root directory")

        # Load GeoJSON data
        geojson_files = [f for f in os.listdir(self.video_geojson_root) 
                        if f.endswith('.geojson')]
        if not geojson_files:
            raise ValueError("No GeoJSON files found in the video root directory")
            
        self.geojson_path = os.path.join(self.video_geojson_root, geojson_files[0])
        with open(self.geojson_path, 'r') as f:
            self.geojson_data = json.load(f)

        # Store crop information from the dataset
        self.crops = None
        self.image_size = None
        # ... existing initialization code ...
        self._cache = {}
        self._cache_size = 100


        # Load class mappings
        import yaml
        with open(os.path.join('mapping/class_mapping.yaml'), 'r') as f:
            mapping_data = yaml.safe_load(f)
            self.class_mapping = mapping_data['class_mapping']
            self.class_names = mapping_data['class_names']

    def _precompute_geometries(self, crop_geometry):
        """Pre-compute and cache all transformed geometries that intersect with the crop box"""
                
        excluded_type_codes = ['wall', 'external-wall', 'glass-wall', 'circulation', 'desk', "workstation-desk", 
                            "break-area-desk", "game-table", "workstation-chair", "meeting-room-chair", "office-equipment-accessories",
                            "couch", "pouchair", "break-area-couch", "waiting-seats", "meeting-table", "meeting-desk", "meeting-room-desk",
                            "floor-outline", "building-outline", "hallway", "vestibule"]

        # First transform features and filter by type_codes and crop_box intersection
        filtered_features = []
        for feature in self.geojson_data['features']:
            # Get typeCode from properties
            type_code = feature.get('properties', {}).get('typeCode')
            
            # Skip if type_code is in the excluded list
            if excluded_type_codes is not None and type_code in excluded_type_codes:
                continue

            # Transform the geometry first
            geom = shape(feature['geometry'])
            transformed_geom = self._transform_geometry(geom)

            # Check if transformed geometry intersects with original crop box
            if crop_geometry.contains(transformed_geom):
                filtered_features.append((feature, transformed_geom))

        # Now do random selection from the filtered features
        import random
        k = 32
        if k is not None and k < len(filtered_features):
            filtered_features = random.sample(filtered_features, k)

        # Create final dictionary of transformed geometries
        transformed_geoms = {}
        feature_classes = {}
        for obj_id, (feature, transformed_geom) in enumerate(filtered_features, start=1):
            transformed_geoms[obj_id] = transformed_geom
            type_code = feature.get('properties', {}).get('typeCode')  # Get type_code here
            feature_classes[obj_id] = self.class_mapping.get(type_code, 0)
        
        return transformed_geoms, feature_classes


    def set_crop_info(self, crops, image_size):
        """Store crop information for later use"""
        self.crops = crops
        self.image_size = image_size

    def _transform_coordinates(self, coords):
        """Transforms coordinates from DXF space to image space."""
        coords_homogeneous = np.ones((len(coords), 3))
        coords_homogeneous[:, :2] = coords
        transformed_coords = np.dot(self.transform_matrix, coords_homogeneous.T).T
        return transformed_coords[:, :2]

    def _transform_geometry(self, geom):
        """Transform a geometry from DXF space to image space"""
        if geom.is_empty:
            return geom
        
        if geom.geom_type == 'Polygon':
            exterior_coords = np.array(geom.exterior.coords)
            transformed_exterior = self._transform_coordinates(exterior_coords)
            
            interiors = []
            for interior in geom.interiors:
                interior_coords = np.array(interior.coords)
                transformed_interior = self._transform_coordinates(interior_coords)
                interiors.append(transformed_interior)
                
            return self._create_polygon(transformed_exterior, interiors)
            
        elif geom.geom_type == 'MultiPolygon':
            transformed_polygons = [self._transform_geometry(poly) for poly in geom.geoms]
            return unary_union(transformed_polygons)
        
        return geom

    def load(self, frame_id):
        """Optimized load function for large images"""
        if frame_id in self._cache:
            return self._cache[frame_id]

        if self.crops is None or self.image_size is None:
            raise ValueError("Crop information not set. Call set_crop_info first.")

        crop_box = self.crops[frame_id]
        binary_segments = {}
        segment_classes = {}

        ## Create crop box geometry for intersection testing
        crop_geometry = box(crop_box[0], crop_box[1], crop_box[2], crop_box[3])
        crop_geometry = box(
        crop_box[0],  # left = minx
        crop_box[3],  # bottom = miny
        crop_box[2],  # right = maxx
        crop_box[1]   # top = maxy
        )
        
        # Calculate crop dimensions
        crop_width = crop_box[2] - crop_box[0]
        crop_height = crop_box[3] - crop_box[1]
        
        # Pre-compute and cache transformed geometries
        self._transformed_geometries, self._feature_classes = self._precompute_geometries(crop_geometry)
        
        def process_object(obj_id):
            """Process single object - suitable for parallel processing"""
            geom = self._transformed_geometries[obj_id]
            cls = self._feature_classes[obj_id]

            # # Quick intersection test
            # if not geom.intersects(crop_geometry):
            #     return None
                
            # Create mask only for the crop region
            mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
            
            # Convert geometry to pixel coordinates relative to crop
            geom = geom.buffer(0)
            geom_cropped = geom.intersection(crop_geometry)
            if geom_cropped.is_empty:
                return None
                
            # Convert to pixel coordinates and draw
            try:
                points = []
                if geom_cropped.geom_type == 'Polygon':
                    points = [geom_cropped.exterior.coords]
                elif geom_cropped.geom_type == 'MultiPolygon':
                    points = [poly.exterior.coords for poly in geom_cropped.geoms]
                
                for poly_points in points:
                    # Convert to relative coordinates
                    poly_points = np.array(poly_points)
                    poly_points[:, 0] -= crop_box[0]
                    poly_points[:, 1] -= crop_box[1]
                    
                    # Convert to integers
                    poly_points = poly_points.astype(np.int32)
                    
                    # Draw polygon using cv2
                    cv2.fillPoly(mask, [poly_points], 1)
                
                # Convert holes if any
                if geom_cropped.geom_type == 'Polygon':
                    for interior in geom_cropped.interiors:
                        points = np.array(interior.coords)
                        points[:, 0] -= crop_box[0]
                        points[:, 1] -= crop_box[1]
                        cv2.fillPoly(mask, [points.astype(np.int32)], 0)
                
                # Only return if mask contains any positive pixels
                if mask.any():
                    return obj_id, torch.from_numpy(mask.astype(bool)), cls
                    
            except Exception as e:
                print(f"Error processing object {obj_id}: {e}")
                return None
                
            return None

        # Process objects in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(process_object, self._transformed_geometries.keys())
            
        # Collect results
        for result in results:
            if result is not None:
                obj_id, mask, cls = result
                binary_segments[obj_id] = mask
                segment_classes[obj_id] = cls

        # Cache result
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[frame_id] = (binary_segments, segment_classes)

        return binary_segments, segment_classes

    def _create_polygon(self, exterior, interiors):
        """Helper function to create a polygon from coordinates"""
        from shapely.geometry import Polygon
        return Polygon(exterior, interiors)
    
    def __len__(self):
        return len(self.crops) if self.crops is not None else 0
    
    # def visualize_transformed_geometries(self, crop_geometry):
    # """
    # Visualize all transformed geometries and crop box for debugging.
    # Blue: Transformed geometries
    # Red: Crop box
    # Green: Selected geometries (after filtering and random selection)
    # """
    # import matplotlib.pyplot as plt
    # from matplotlib.patches import Polygon as MplPolygon
    # from shapely.geometry import MultiPolygon, Polygon, Point, LineString, MultiPoint, MultiLineString
    
    # # Create figure and axis
    # fig, ax = plt.subplots(figsize=(15, 15))
    
    # def plot_polygon(geometry, **kwargs):
    #     """Helper function to plot a geometry"""
    #     if isinstance(geometry, (Point, MultiPoint)):
    #         # Skip points for now
    #         return
    #     elif isinstance(geometry, (LineString, MultiLineString)):
    #         # Skip lines for now
    #         return
    #     elif isinstance(geometry, MultiPolygon):
    #         for p in geometry.geoms:
    #             if isinstance(p, Polygon):
    #                 plot_polygon(p, **kwargs)
    #     elif isinstance(geometry, Polygon):
    #         # Extract exterior coordinates
    #         exterior_coords = np.array(geometry.exterior.coords)
    #         patch = MplPolygon(exterior_coords, **kwargs)
    #         ax.add_patch(patch)
            
    #         # Plot interior rings (holes)
    #         for interior in geometry.interiors:
    #             interior_coords = np.array(interior.coords)
    #             patch = MplPolygon(interior_coords, **kwargs)
    #             ax.add_patch(patch)
    
    # # Plot all transformed geometries in blue (light blue with alpha)
    # for feature in self.geojson_data['features']:
    #     geom = shape(feature['geometry'])
    #     transformed_geom = self._transform_geometry(geom)
    #     plot_polygon(transformed_geom, facecolor='blue', edgecolor='blue', alpha=0.2)
    
    # # Plot crop box in red
    # plot_polygon(crop_geometry, facecolor='none', edgecolor='red', linewidth=2)
    
    # # Get and plot selected geometries
    # selected_geoms = self._precompute_geometries(crop_geometry)
    # for geom in selected_geoms.values():
    #     plot_polygon(geom, facecolor='green', edgecolor='green', alpha=0.5)
    
    # # Set axis limits with some padding
    # all_bounds = [geom.bounds for geom in selected_geoms.values()]
    # all_bounds.append(crop_geometry.bounds)
    
    # min_x = min(bound[0] for bound in all_bounds)
    # min_y = min(bound[1] for bound in all_bounds)
    # max_x = max(bound[2] for bound in all_bounds)
    # max_y = max(bound[3] for bound in all_bounds)
    
    # padding = (max(max_x - min_x, max_y - min_y)) * 0.1
    # ax.set_xlim(min_x - padding, max_x + padding)
    # ax.set_ylim(min_y - padding, max_y + padding)
    
    # # Add legend
    # from matplotlib.patches import Patch
    # legend_elements = [
    #     Patch(facecolor='blue', alpha=0.2, label='All Transformed Geometries'),
    #     Patch(facecolor='none', edgecolor='red', label='Crop Box'),
    #     Patch(facecolor='green', alpha=0.5, label='Selected Geometries')
    # ]
    # ax.legend(handles=legend_elements)
    
    # # Add title and grid
    # plt.title('Transformed Geometries and Crop Box Visualization')
    # plt.grid(True)
    # plt.axis('equal')
    
    # # Show plot
    # plt.savefig('./crop_box.png')

if __name__ == '__main__':

    # Initialize loader
    loader = GeoJSONSegmentLoader(
        video_geojson_root='/ssd/datasets/sam2_unit_geojson_dataset_jan3/geojsons/3070_5372'
    )

    # Load segments and classes
    segments, classes = loader.load(0)

    # Access segments and their classes
    for obj_id in segments:
        mask = segments[obj_id]
        class_id = classes[obj_id]
        class_name = loader.get_class_name(class_id)
        print(f"Object {obj_id}: Class = {class_name} (ID: {class_id})")