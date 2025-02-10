import torch
from torch import nn
from torch.functional import F
import numpy as np
import json
import os
import cv2
import re
import importlib
from copy import deepcopy
from transformers import AutoImageProcessor, Dinov2Model

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
        
        # DINOv2 initialization
        self.feature_extractor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.dinov2 = Dinov2Model.from_pretrained("facebook/dinov2-base")
        self.feature_dim = 768  # DINOv2 base features dimension
        
        # Projection layer for feature adaptation
        self.proj = nn.Linear(self.feature_dim, 512, bias=False)
        nn.init.eye_(self.proj.weight)
        
        # Original parameters
        self.stride = stride
        self.kernel_size = kernel_size
        self.normalize = normalize
        self.dataset = dataset
        self.visualize = visualize
        self.remove_background = remove_background
        self.topk = 3
        self.m = 0.7
        self.split = split
        self.setting = setting
        self.video_path = video_path
        self.avg_features_path = "./data/Thumos14/support_videos_features/"
        
        # TTA parameters
        self.tta_momentum = 0.95
        self.confidence_threshold = 0.7
        
        # Dataset initialization
        if self.dataset == "thumos":
            dict_test_name = (
                f"t2_dict_test_thumos_{split}"
                if self.setting == 50
                else f"t1_dict_test_thumos_{split}" if self.setting == 75 else None
            )
            self.annotations_path = "./data/thumos_annotations/thumos_anno_action.json"
            self.video_dir = os.path.join(self.video_path, "Thumos14/videos/")
        elif self.dataset == "anet":
            dict_test_name = (
                f"t2_dict_test_{split}"
                if self.setting == 50
                else f"t1_dict_test_{split}" if self.setting == 75 else None
            )
            self.annotations_path = "./data/activitynet_annotations/anet_anno_action.json"
            self.video_dir = os.path.join(self.video_path, "ActivityNetVideos/videos/")
        else:
            raise ValueError(f"Not implemented dataset: {self.dataset}")

        self.dict_test = getattr(
            importlib.import_module("config.zero_shot"), dict_test_name, None
        )
        
        self.cls_names = self.dict_test
        self.num_classes = len(self.cls_names)
        self.inverted_cls = {v: k for k, v in self.cls_names.items()}
        
        with open(self.annotations_path, "r") as f:
            self.annotations = json.load(f)
            
        self.avg_features = self.load_avg_features(self.avg_features_path)
        self.original_avg_features = self.avg_features.clone()
        self.original_proj_weight = self.proj.weight.data.clone()



    def load_avg_features(self, path):
        """Load precomputed DINOv2 features for each class"""
        avg_features = []
        for class_name in self.cls_names:
            feature_file = os.path.join(path, class_name,f"{class_name}_average.npy")
            print(f"Loading features for {class_name} from {feature_file}")
            if os.path.exists(feature_file):
                feat = np.load(feature_file)
                feat = torch.from_numpy(feat).float()
                avg_features.append(self.proj(feat).mean(0))
            else:
                raise FileNotFoundError(f"Feature file missing for {class_name}")
        return torch.stack(avg_features)

    def adapt_features(self, features, device):
        """TTA with dual adaptation (prototypes + projection)"""
        # Normalize and project features
        features = F.normalize(features, dim=-1)
        projected = self.proj(features)
        projected = F.normalize(projected, dim=-1)
        
        # Calculate similarities
        similarities = projected @ self.avg_features.to(device).T
        confidences = F.softmax(similarities / 0.1, dim=-1)
        
        # Get high-confidence predictions
        max_conf, preds = torch.max(confidences, dim=-1)
        mask = max_conf > self.confidence_threshold
        
        if mask.any():
            # Update prototypes with momentum
            for cls_idx in range(self.num_classes):
                cls_mask = (preds == cls_idx) & mask
                if cls_mask.any():
                    cls_feats = projected[cls_mask].mean(0)
                    self.avg_features[cls_idx] = (
                        self.tta_momentum * self.avg_features[cls_idx].to(device) +
                        (1 - self.tta_momentum) * cls_feats
                    )
                    self.avg_features[cls_idx] = F.normalize(self.avg_features[cls_idx], dim=-1)
            
            # Update projection matrix
            cov_matrix = features.T @ features
            new_proj = self.tta_momentum * self.proj.weight + \
                      (1 - self.tta_momentum) * cov_matrix[:512]
            self.proj.weight.data.copy_(new_proj)
        
        return projected

    def moving_average(self, data, window_size):
        padding_size = window_size // 2
        padded_data = F.pad(data, (padding_size, padding_size), mode='reflect')
        kernel = torch.ones(window_size).to(data.device) / window_size
        return F.conv1d(padded_data.unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0)).squeeze()

    def select_segments(self, similarity):
        if self.dataset == 'thumos':
            mask = similarity > similarity.mean()
        elif self.dataset == 'anet': 
            mask = similarity > self.m
        else:
            raise ValueError(f"Not implemented dataset: {self.dataset}")
            
        selected = torch.nonzero(mask).squeeze()
        segments = []
        current_segment = [selected[0], selected[0]] if selected.numel() else []
        
        for idx in selected[1:]:
            if idx <= current_segment[1] + self.stride:
                current_segment[1] = idx
            else:
                segments.append(current_segment)
                current_segment = [idx, idx]
        if current_segment:
            segments.append(current_segment)
            
        return segments

    def get_video_fps(self, video_name):
        for ext in [".mp4", ".mkv", ".webm"]:
            video_path = os.path.join(self.video_dir, video_name + ext)
            if os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                return fps
        return 30  # default if video not found

    def forward(self, x):
        idx, video_name, features = x
        video_name = video_name[0]
        features = features.squeeze(0)
        
        # TTA Feature Adaptation
        adapted_features = self.adapt_features(features, features.device)
        
        # Calculate similarities
        similarities = adapted_features @ self.avg_features.to(features.device).T
        class_sims, class_ids = similarities.max(dim=-1)
        
        # Temporal processing
        if self.dataset == "thumos":
            print("class_sims shape", class_sims.shape)
            class_sims = class_sims.unsqueeze(0)
            smoothed_sims = self.moving_average(class_sims, self.kernel_size)
        else:
            smoothed_sims = class_sims
            
        # Segment selection
        segments = self.select_segments(smoothed_sims)
        
        # Prepare output
        output = []
        for seg in segments:
            start, end = seg
            seg_feats = adapted_features[start:end+1].mean(dim=0)
            final_sim = seg_feats @ self.avg_features.to(features.device).T
            cls_id = final_sim.argmax().item()
            
            output.append({
                "label": cls_id,
                "score": final_sim[cls_id].item(),
                "segment": [start.item(), end.item()]
            })
        
        # Reset TTA parameters
        self.avg_features = self.original_avg_features.clone()
        self.proj.weight.data.copy_(self.original_proj_weight.to(features.device))

        # calculate prediction mask
        pred_mask = torch.zeros_like(smoothed_sims)
        for seg in output:
            pred_mask[seg["segment"][0]:seg["segment"][1]+1] = 1

        # calculate ground truth mask
        gt_segments = [
            [int(anno["segment"][0]*self.get_video_fps(video_name)), int(anno["segment"][1]*self.get_video_fps(video_name))]
            for anno in self.annotations[video_name]["annotations"]
        ]
        gt_mask = torch.zeros_like(smoothed_sims)
        
        # calculate unique labels
        unique_labels = list(set(
            anno["label"] for anno in self.annotations[video_name]["annotations"]
        ))
        
        if self.visualize:
            return self._visualize_output(video_name, smoothed_sims, output)
        else:
            return (video_name, output, None, None, None, None)

    def _visualize_output(self, video_name, similarities, output):
        fps = self.get_video_fps(video_name)
        gt_segments = [
            [int(anno["segment"][0]*fps), int(anno["segment"][1]*fps)] 
            for anno in self.annotations[video_name]["annotations"]
        ]
        gt_labels = list(set(
            anno["label"] for anno in self.annotations[video_name]["annotations"]
        ))
        
        fig = plt.figure(figsize=(25, 20))
        plt.plot(similarities.cpu().numpy(), c='darkblue', alpha=0.7)
        for seg in output:
            plt.axvspan(seg["segment"][0], seg["segment"][1], color='red', alpha=0.1)
        for seg in gt_segments:
            plt.axvspan(seg[0], seg[1], color='green', alpha=0.1)
            
        return (video_name, output, None, None, gt_labels, fig)