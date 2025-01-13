# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import os
from dataclasses import dataclass

from typing import List, Optional

import random

import torch

from iopath.common.file_io import g_pathmgr

from omegaconf.listconfig import ListConfig

from training.dataset.vos_segment_loader import (
    JSONSegmentLoader,
    MultiplePNGSegmentLoader,
    PalettisedPNGSegmentLoader,
    SA1BSegmentLoader,
    GeoJSONSegmentLoader,
)


@dataclass
class VOSFrame:
    frame_idx: int
    image_path: str
    data: Optional[torch.Tensor] = None
    is_conditioning_only: Optional[bool] = False


@dataclass
class VOSVideo:
    video_name: str
    video_id: int
    frames: List[VOSFrame]

    def __len__(self):
        return len(self.frames)


class VOSRawDataset:
    def __init__(self):
        pass

    def get_video(self, idx):
        raise NotImplementedError()


class PNGRawDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        is_palette=False,
        single_object_mode=False,
        truncate_video=-1,
        frames_sampling_mult=False,
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.sample_rate = sample_rate
        self.is_palette = is_palette
        self.single_object_mode = single_object_mode
        self.truncate_video = truncate_video

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files
        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

        if self.single_object_mode:
            # single object mode
            self.video_names = sorted(
                [
                    os.path.join(video_name, obj)
                    for video_name in self.video_names
                    for obj in os.listdir(os.path.join(self.gt_folder, video_name))
                ]
            )

        if frames_sampling_mult:
            video_names_mult = []
            for video_name in self.video_names:
                num_frames = len(os.listdir(os.path.join(self.img_folder, video_name)))
                video_names_mult.extend([video_name] * num_frames)
            self.video_names = video_names_mult

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]

        if self.single_object_mode:
            video_frame_root = os.path.join(
                self.img_folder, os.path.dirname(video_name)
            )
        else:
            video_frame_root = os.path.join(self.img_folder, video_name)

        video_mask_root = os.path.join(self.gt_folder, video_name)

        if self.is_palette:
            segment_loader = PalettisedPNGSegmentLoader(video_mask_root)
        else:
            segment_loader = MultiplePNGSegmentLoader(
                video_mask_root, self.single_object_mode
            )

        all_frames = sorted(glob.glob(os.path.join(video_frame_root, "*.png")))
        if self.truncate_video > 0:
            all_frames = all_frames[: self.truncate_video]
        frames = []
        for _, fpath in enumerate(all_frames[:: self.sample_rate]):
            fid = int(os.path.basename(fpath).split(".")[0])
            frames.append(VOSFrame(fid, image_path=fpath))
        video = VOSVideo(video_name, idx, frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)


class SA1BRawDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        num_frames=1,
        mask_area_frac_thresh=1.1,  # no filtering by default
        uncertain_iou=-1,  # no filtering by default
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.num_frames = num_frames
        self.mask_area_frac_thresh = mask_area_frac_thresh
        self.uncertain_iou = uncertain_iou  # stability score

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)
            subset = [
                path.split(".")[0] for path in subset if path.endswith(".png")
            ]  # remove extension

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files and it exists
        self.video_names = [
            video_name for video_name in subset if video_name not in excluded_files
        ]

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]

        video_frame_path = os.path.join(self.img_folder, video_name + ".png")
        video_mask_path = os.path.join(self.gt_folder, video_name + ".json")

        segment_loader = SA1BSegmentLoader(
            video_mask_path,
            mask_area_frac_thresh=self.mask_area_frac_thresh,
            video_frame_path=video_frame_path,
            uncertain_iou=self.uncertain_iou,
        )

        frames = []
        for frame_idx in range(self.num_frames):
            frames.append(VOSFrame(frame_idx, image_path=video_frame_path))
        video_name = video_name.split("_")[-1]  # filename is sa_{int}
        # video id needs to be image_id to be able to load correct annotation file during eval
        video = VOSVideo(video_name, int(video_name), frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)


class JSONRawDataset(VOSRawDataset):
    """
    Dataset where the annotation in the format of SA-V json files
    """

    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        rm_unannotated=True,
        ann_every=1,
        frames_fps=24,
    ):
        self.gt_folder = gt_folder
        self.img_folder = img_folder
        self.sample_rate = sample_rate
        self.rm_unannotated = rm_unannotated
        self.ann_every = ann_every
        self.frames_fps = frames_fps

        # Read and process excluded files if provided
        excluded_files = []
        if excluded_videos_list_txt is not None:
            if isinstance(excluded_videos_list_txt, str):
                excluded_videos_lists = [excluded_videos_list_txt]
            elif isinstance(excluded_videos_list_txt, ListConfig):
                excluded_videos_lists = list(excluded_videos_list_txt)
            else:
                raise NotImplementedError

            for excluded_videos_list_txt in excluded_videos_lists:
                with open(excluded_videos_list_txt, "r") as f:
                    excluded_files.extend(
                        [os.path.splitext(line.strip())[0] for line in f]
                    )
        excluded_files = set(excluded_files)

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

    def get_video(self, video_idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[video_idx]
        video_json_path = os.path.join(self.gt_folder, video_name + "_manual.json")
        segment_loader = JSONSegmentLoader(
            video_json_path=video_json_path,
            ann_every=self.ann_every,
            frames_fps=self.frames_fps,
        )

        frame_ids = [
            int(os.path.splitext(frame_name)[0])
            for frame_name in sorted(
                os.listdir(os.path.join(self.img_folder, video_name))
            )
        ]

        frames = [
            VOSFrame(
                frame_id,
                image_path=os.path.join(
                    self.img_folder, f"{video_name}/%05d.png" % (frame_id)
                ),
            )
            for frame_id in frame_ids[:: self.sample_rate]
        ]

        if self.rm_unannotated:
            # Eliminate the frames that have not been annotated
            valid_frame_ids = [
                i * segment_loader.ann_every
                for i, annot in enumerate(segment_loader.frame_annots)
                if annot is not None and None not in annot
            ]
            frames = [f for f in frames if f.frame_idx in valid_frame_ids]

        video = VOSVideo(video_name, video_idx, frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)

import os
import glob
import json
import numpy as np
from shapely.geometry import shape
from shapely.affinity import scale
from PIL import Image, ImageDraw

class GeoJSONDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        crop_levels=2,
        overlap=0.5,
        file_list_txt=None,
        truncate_video=0,
        sample_rate=1,
        single_object_mode=False,
    ):
        self.gt_folder = gt_folder
        self.img_folder = img_folder
        self.crop_levels = crop_levels
        self.overlap = overlap
        self.truncate_video = truncate_video
        self.sample_rate = sample_rate
        self.single_object_mode = single_object_mode

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = [
                os.path.splitext(img_file)[0]
                for img_file in os.listdir(img_folder)
                if img_file.endswith(".png")
            ]
        
        self.video_names = sorted(subset)

    def __len__(self):
        return len(self.video_names)
    
    def _generate_crops(self, image, level):
        """Generates cropped images at a given level."""
        w, h = image.size
        crop_h = h // (2 ** level)
        crop_w = w // (2 ** level)
        
        step_h = int(crop_h * (1 - self.overlap))
        step_w = int(crop_w * (1 - self.overlap))
        
        # Ensure minimum step size
        step_h = max(step_h, 1)
        step_w = max(step_w, 1)

        crops = []
        for top in range(0, h - crop_h + step_h, step_h):
            for left in range(0, w - crop_w + step_w, step_w):
                # Adjust last crops to ensure they don't exceed image boundaries
                bottom = min(top + crop_h, h)
                right = min(left + crop_w, w)
                
                # Adjust start positions for last crops to maintain consistent crop size
                if bottom - top < crop_h:
                    top = max(0, h - crop_h)
                if right - left < crop_w:
                    left = max(0, w - crop_w)
                    
                crops.append((left, top, right, bottom))
        return crops
            
    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]
        image_path = os.path.join(self.img_folder, f"{video_name}.png")
        video_geojson_root = os.path.join(self.gt_folder, video_name)

        image = Image.open(image_path).convert("RGB") 
        segment_loader = GeoJSONSegmentLoader(video_geojson_root)

        # Generate all crops first
        all_crops = []
        for level in range(self.crop_levels):
            crops = self._generate_crops(image, level)
            all_crops.extend(crops)

        # Set crop information in the segment loader
        segment_loader.set_crop_info(all_crops, image.size)

        # Create a list of indices and shuffle them
        crop_indices = list(range(len(all_crops)))
        random.shuffle(crop_indices)

        # Create frames with shuffled order
        frames = []
        for frame_idx, crop_idx in enumerate(crop_indices):
            crop_box = all_crops[crop_idx]
            crop = image.crop(crop_box)
            # Convert to tensor
            crop_tensor = torch.from_numpy(np.array(crop) / 255.0).permute(2, 0, 1)

            frame = VOSFrame(
                frame_idx=crop_idx,  # Keep original crop_idx for segment_loader
                image_path=image_path,
                data=crop_tensor,
                is_conditioning_only=False
            )
            frames.append(frame)

        video = VOSVideo(video_name, idx, frames)
        return video, segment_loader


def visualize_batch(batch, save_path="batch_viz.png"):
    import matplotlib.pyplot as plt
    import numpy as np
    
    # ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Get the image and masks
    image = batch.img_batch.cpu().numpy()[0].transpose(1, 2, 0)  # [H, W, 3]
    masks = batch.masks.cpu().numpy()
    
    # Denormalize image
    image = image * std[None, None, :] + mean[None, None, :]
    #image = np.clip(image, 0, 1)  # Clip values to valid range
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
    
    # Generate distinct colors using HSV color space
    num_masks = masks.shape[0]
    hsv_colors = np.zeros((num_masks, 3))
    hsv_colors[:, 0] = np.linspace(0, 1, num_masks)  # Hue
    hsv_colors[:, 1] = 0.8  # Saturation
    hsv_colors[:, 2] = 0.9  # Value
    
    # Convert HSV to RGB
    from matplotlib.colors import hsv_to_rgb
    rgb_colors = hsv_to_rgb(hsv_colors)
    
    # Create masked visualization
    mask_vis = np.zeros((*masks.shape[1:], 3))
    for idx in range(num_masks):
        mask = masks[idx]
        if mask.any():
            mask_vis[mask] = rgb_colors[idx]
    
    # Plot
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(mask_vis)
    ax2.set_title(f'All Masks (Total: {num_masks})')
    ax2.axis('off')
    
    ax3.imshow(image)
    ax3.imshow(mask_vis, alpha=0.5)
    ax3.set_title('Overlay')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
# visualize_batch(batch[0], save_path="batch_viz.png")
