import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import cv2
import re
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import List, Dict
import importlib

class T3AL0Net(nn.Module):
    def __init__(
        self,
        stride: int,
        kernel_size: int,
        normalize: bool,
        dataset: str,
        visualize: bool,
        remove_background: bool,
        split: int,
        setting: int, 
        video_path: str,
    ):
        super(T3AL0Net, self).__init__()
        
        # Original parameters (preserved exactly)
        self.stride = stride
        self.kernel_size = kernel_size
        self.normalize = normalize
        self.dataset = dataset
        self.visualize = visualize
        self.remove_background = remove_background
        self.split = split
        self.setting = setting
        self.video_path = video_path
        self.topk = 3
        self.m = 0.7
        
        # Original CLIP-related code removed
        # TTA parameters preserved from original
        self.tta_momentum = 0.9
        self.confidence_threshold = 0.7
        
        # New DINOv2 compatibility layer (hidden implementation detail)
        self._feature_dim = 1024  # DINOv2 ViT-L/14 feature dimension
        self._prototype_dim = 768  # Matching original CLIP text dimension
        self.proj = nn.Parameter(torch.empty(self._feature_dim, self._prototype_dim))
        nn.init.kaiming_normal_(self.proj)
        
        # Original dataset configuration
        if self.dataset == "thumos":
            dict_test_name = (
                f"t2_dict_test_thumos_{split}"
                if self.setting == 50
                else f"t1_dict_test_thumos_{split}" if self.setting == 75 else None
            )
            self.annotations_path = "./data/thumos_annotations/thumos_anno_action.json"
            self.video_dir = os.path.join(self.video_path, "Thumos14/videos/")
            self.avg_features_path = "./data/Thumos14/support_videos_features/"
        elif self.dataset == "anet":
            dict_test_name = (
                f"t2_dict_test_{split}"
                if self.setting == 50
                else f"t1_dict_test_{split}" if self.setting == 75 else None
            )
            self.annotations_path = "./data/activitynet_annotations/anet_anno_action.json"
            self.video_dir = os.path.join(self.video_path, "ActivityNetVideos/videos/")
            self.avg_features_path = "./data/ActivityNet/support_videos_features/"
        else:
            raise ValueError(f"Not implemented dataset: {self.dataset}")

        # Original config loading
        self.dict_test = getattr(
            importlib.import_module("config.zero_shot"), dict_test_name, None
        )
        self.cls_names = self.dict_test
        self.num_classes = len(self.cls_names)
        self.inverted_cls = {v: k for k, v in self.cls_names.items()}
        
        # Original annotation loading
        with open(self.annotations_path, "r") as f:
            self.annotations = json.load(f)
            
        # Original feature loading (now for DINOv2 features)
        self.avg_features = self.load_avg_features(self.avg_features_path)
        self.original_avg_features = deepcopy(self.avg_features)
    def _init_dataset_config(self):
        if self.dataset == "thumos":
            self.annotations_path = "./data/thumos_annotations/thumos_anno_action.json"
            self.video_dir = os.path.join(self.video_path, "Thumos14/videos/")
            self.prototype_path = "./data/Thumos14/support_videos_features/"
        elif self.dataset == "anet":
            self.annotations_path = "./data/activitynet_annotations/anet_anno_action.json"
            self.video_dir = os.path.join(self.video_path, "ActivityNetVideos/videos/")
            self.prototype_path = "./data/ActivityNet/support_videos_features/"
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

    def _load_annotations(self):
        with open(self.annotations_path, "r") as f:
            self.annotations = json.load(f)
            
    def _init_prototypes(self):
        """Initialize class prototypes from precomputed features"""
        self.prototypes = {}
        for cls_name in os.listdir(self.prototype_path):
            cls_dir = os.path.join(self.prototype_path, cls_name)
            if os.path.isdir(cls_dir):
                proto_file = os.path.join(cls_dir, f"{cls_name}_average.npy")
                if os.path.exists(proto_file):
                    proto = np.load(proto_file)
                    self.prototypes[cls_name] = torch.from_numpy(proto).float().mean(0)
        
        self.cls_names = list(self.prototypes.keys())
        self.num_classes = len(self.cls_names)
        self.original_prototypes = torch.stack(list(self.prototypes.values()))
        self.prototypes = self.original_prototypes.clone()

    def _project_features(self, features: torch.Tensor) -> torch.Tensor:
        """Project features to prototype space with normalization"""
        projected = F.normalize(features @ self.proj, dim=-1)
        if self.remove_background:
            projected -= projected.mean(dim=0)
        return F.normalize(projected, dim=-1) if self.normalize else projected

    def _adapt_parameters(self, features: torch.Tensor):
        """Test-time adaptation of projection and prototypes"""
        with torch.no_grad():
            # Get initial predictions
            similarities = features @ self.prototypes.T
            confidences = F.softmax(similarities / 0.07, dim=-1)
            max_conf, preds = torch.max(confidences, dim=-1)
            
            # Update prototypes
            valid = max_conf > self.confidence_thresh
            for cls_idx in torch.unique(preds[valid]):
                cls_mask = (preds == cls_idx) & valid
                if cls_mask.any():
                    cls_feats = features[cls_mask].mean(0)
                    self.prototypes[cls_idx] = (
                        self.tta_momentum * self.prototypes[cls_idx] +
                        (1 - self.tta_momentum) * cls_feats
                    )
                    self.prototypes[cls_idx] = F.normalize(self.prototypes[cls_idx], dim=-1)

            # Update projection matrix
            grad = features.T @ (self.prototypes[preds[valid]] - features[valid])
            self.proj.data = self.tta_momentum * self.proj + (1 - self.tta_momentum) * grad.T
            self.proj.data = F.normalize(self.proj.data, dim=-1)

    def _temporal_smoothing(self, similarities: torch.Tensor) -> torch.Tensor:
        """Apply temporal smoothing to similarity scores"""
        pad = self.kernel_size // 2
        return F.avg_pool1d(
            similarities.unsqueeze(0),
            kernel_size=self.kernel_size,
            padding=pad,
            stride=self.stride
        ).squeeze(0)

    def _detect_segments(self, similarities: torch.Tensor) -> List[List[int]]:
        """Detect action segments from similarity scores"""
        threshold = similarities.mean() if self.dataset == 'thumos' else self.m
        mask = similarities > threshold
        segments = []
        start_idx = None
        
        for i, val in enumerate(mask):
            if val and start_idx is None:
                start_idx = i
            elif not val and start_idx is not None:
                segments.append([start_idx, i-1])
                start_idx = None
        if start_idx is not None:
            segments.append([start_idx, len(mask)-1])
            
        return segments

    def forward(self, x: tuple) -> tuple:
        # Unpack input tuple
        idx, video_name, features = x
        video_name = video_name[0]
        features = features.squeeze(0)
        
        # Store original parameters
        original_proj = self.proj.clone()
        original_prototypes = self.prototypes.clone()
        
        # Main processing pipeline
        projected = self._project_features(features)
        self._adapt_parameters(projected)
        projected = self._project_features(features)  # Re-project with updated params
        
        # Calculate final similarities
        similarities = projected @ self.prototypes.T
        similarities = self._temporal_smoothing(similarities.mean(dim=-1))
        
        # Detect segments
        segments = self._detect_segments(similarities)
        
        # Create output format matching original
        output = []
        pred_mask = torch.zeros(len(similarities), device=features.device)
        for seg in segments:
            output.append({
                'label': similarities[seg[0]:seg[1]].argmax().item(),
                'score': similarities[seg[0]:seg[1]].mean().item(),
                'segment': [seg[0], seg[1]]
            })
            pred_mask[seg[0]:seg[1]] = 1
            
        # Get ground truth information
        gt_info = self.annotations.get(video_name, {})
        gt_segments = [anno['segment'] for anno in gt_info.get('annotations', [])]
        fps = self._get_video_fps(video_name)
        gt_mask = torch.zeros(len(similarities), device=features.device)
        for seg in gt_segments:
            start = int(seg[0] * fps)
            end = int(seg[1] * fps)
            gt_mask[start:end] = 1
            
        unique_labels = list(set(
            anno['label'] for anno in gt_info.get('annotations', [])
        ))
        
        # Reset parameters for next video
        self.proj.data = original_proj
        self.prototypes.data = original_prototypes
        
        # Visualization
        plt_obj = None
        if self.visualize:
            plt_obj = self._create_visualization(
                video_name, similarities, output, gt_segments, unique_labels
            )

        return (
            video_name,
            output,
            pred_mask,
            gt_mask,
            unique_labels,
            plt_obj
        )

    def _get_video_fps(self, video_name: str) -> float:
        for ext in ['.mp4', '.avi', '.mkv']:
            path = os.path.join(self.video_dir, f"{video_name}{ext}")
            if os.path.exists(path):
                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                return fps
        return 30.0  # Default fallback

    def _create_visualization(self, video_name: str, similarities: torch.Tensor,
                             output: List[dict], gt_segments: list, unique_labels: list) -> plt.Figure:
        fig = plt.figure(figsize=(25, 20))
        plt.scatter(
            torch.arange(len(similarities)),
            similarities.cpu().numpy(),
            c='darkblue',
            s=1,
            alpha=0.5
        )
        plt.title(video_name)
        
        # Add predicted labels
        for i, seg in enumerate(output):
            plt.text(
                0.7, 0.9 - i*0.05,
                f"{self.cls_names[seg['label']]}",
                fontsize=20,
                transform=plt.gcf().transFigure,
                c='red'
            )
            
        # Add ground truth labels
        for i, label in enumerate(unique_labels):
            plt.text(
                0.05, 0.9 - i*0.05,
                label,
                fontsize=20,
                transform=plt.gcf().transFigure,
                c='green'
            )
            
        # Draw segments
        for seg in gt_segments:
            plt.axvspan(seg[0], seg[1], color='green', alpha=0.2)
        for seg in output:
            plt.axvspan(seg['segment'][0], seg['segment'][1], color='red', alpha=0.1)
            
        return fig