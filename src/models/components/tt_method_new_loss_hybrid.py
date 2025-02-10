import torch
from torch import nn
from torch.functional import F
from src.models.components.loss import ByolLoss, HybridLoss
import open_clip
import matplotlib.pyplot as plt
import json
import os
import cv2
import re
import copy
import importlib
import numpy as np

tokenize = open_clip.get_tokenizer("coca_ViT-L-14")

class T3ALNet(nn.Module):
    def __init__(
        self,
        p: float,
        stride: int,
        randper: int,
        kernel_size: int,
        n: int,
        normalize: bool,
        dataset: str,
        visualize: bool,
        text_projection: bool,
        text_encoder: bool,
        image_projection: bool,
        logit_scale: bool,
        remove_background: bool,
        ltype: str,
        steps: int,
        refine_with_captions: bool,
        split: int,
        setting: int,
        video_path: str,
        avg_features_path: str,
    ):
        super(T3ALNet, self).__init__()

        # Initialize parameters
        self.stride = stride
        self.randper = randper
        self.p = p
        self.n = n
        self.normalize = normalize
        self.text_projection = text_projection
        self.text_encoder = text_encoder
        self.image_projection = image_projection
        self.logit_scale = logit_scale
        self.remove_background = remove_background
        self.ltype = "HY"
        self.steps = 60
        self.refine_with_captions = refine_with_captions
        self.split = 50
        self.setting = setting
        self.dataset = dataset
        self.visualize = visualize
        self.kernel_size = kernel_size
        self.video_path = video_path
        self.avg_features_path = avg_features_path
        self.topk = 3
        self.m = 0.7

        # Load the COCA model
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14", pretrained="mscoco_finetuned_laion2B-s13B-b90k"
        )
        self.model = self.model.float()
        print(f"Loaded COCA model")

        # Dataset configuration
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
            self.video_dir = os.path.join(self.video_path, "ActivityNet/videos/")
        else:
            raise ValueError(f"Not implemented dataset: {self.dataset}")

        # Load class names and average features
        self.dict_test = getattr(
            importlib.import_module("config.zero_shot"), dict_test_name, None
        )
        self.cls_names = self.dict_test
        print(f"Loaded {len(self.cls_names)} classes for zero-shot learning.")
        self.num_classes = len(self.cls_names)
        self.inverted_cls = {v: k for k, v in self.cls_names.items()}
        
        # Load average features
        self.avg_features = self.load_avg_features(self.avg_features_path)

        with open(self.annotations_path, "r") as f:
            self.annotations = json.load(f)

        self.temperature = 0.1
        
        if self.ltype == "BCE":
            self.tta_loss = torch.nn.BCEWithLogitsLoss()
        elif "BYOL" in self.ltype:
            self.tta_loss = ByolLoss()
        elif self.ltype == "HY":
            self.hybrid_loss = HybridLoss()
        else:
            raise ValueError(f"Not implemented loss type: {self.ltype}")

    def load_avg_features(self, path):
        """Load precomputed average features for classes."""
        avg_features = {}
        if not os.path.exists(path):
            raise FileNotFoundError(f"Average features folder not found: {path}")
        
        for class_name in self.cls_names.keys():
            print(f"Loading average features for class {class_name}")
            class_folder_path = os.path.join(path, class_name)
            avg_file_path = os.path.join(class_folder_path, class_name + "_average.npy")
            
            if os.path.exists(avg_file_path):
                feature_array = np.load(avg_file_path)
                avg_features[class_name] = torch.tensor(feature_array, dtype=torch.float32)
            else:
                print(f"Warning: Average feature file not found for class {class_name}")
        
        print(f"Loaded {len(avg_features)} class-specific average features")
        
        # Convert to tensor and average features
        avg_features_tensor = torch.stack([feature.mean(dim=0) for feature in avg_features.values()])
        return avg_features_tensor

    def compute_score(self, x, avg_features):
        """Compute similarity scores between input features and average features."""
        # Normalize features
        x = x / x.norm(dim=-1, keepdim=True)
        avg_features = avg_features / avg_features.norm(dim=-1, keepdim=True)
        
        # Set temperature scaling
        temperature = 0.3
        
        with torch.no_grad():
            # Calculate similarity scores
            dot_product = (x @ avg_features.T)
            scores = dot_product / temperature
            
        # Get predicted class
        pred = scores.argmax(dim=-1)
        return pred, scores

    def infer_pseudo_labels(self, image_features):
        """Infer pseudo-labels using average features."""
        if image_features is None or image_features.numel() == 0:
            raise ValueError("image_features is empty or None")

        # Average the image features
        image_features_avg = image_features.mean(dim=0)
        self.background_embedding = image_features_avg.unsqueeze(0)

        # Check average features
        if self.avg_features is None or self.avg_features.numel() == 0:
            raise ValueError("avg_features is empty or not properly loaded")

        # Move average features to same device
        self.avg_features = self.avg_features.to(image_features_avg.device)
        
        # Compute scores using average features
        pred, scores = self.compute_score(image_features_avg.unsqueeze(0), self.avg_features)
        
        # Select top-k scores
        _, indices = torch.topk(scores, self.topk)
        return indices[0][0], scores

    def forward(self, x, optimizer):
        idx, video_name, image_features_pre = x
        # Use clone instead of deepcopy to maintain gradient flow
        image_features_pre = image_features_pre.clone()
        image_features_pre.requires_grad = True
        
        video_name = video_name[0]
        fps = self.get_video_fps(video_name)

        if not self.image_projection:
            image_features = image_features_pre
            image_features = image_features.squeeze(0)
        else:
            with torch.no_grad():
                image_features = image_features_pre @ self.model.visual.proj
                image_features = image_features.squeeze(0)
                
        indexes, _ = self.infer_pseudo_labels(image_features)
        class_label = self.inverted_cls[indexes.item()]

        segments_gt, unique_labels = self.get_segments_gt(video_name, fps)

        # Configure optimizer with appropriate learning rate
        optimizer = torch.optim.Adam([image_features_pre], lr=1e-3)

        for step in range(self.steps):
            if self.image_projection:
                image_features = (image_features_pre @ self.model.visual.proj).squeeze(0)
                before_optimization_parameters_image_encoder = copy.deepcopy(
                    self.model.visual.state_dict()
                )
                before_optimization_image_projection = copy.deepcopy(
                    self.model.visual.proj
                )

            # Get the average feature for the predicted class
            class_avg_feature = self.avg_features[indexes.item()].to(image_features.device)
            
            features = image_features - self.background_embedding if self.remove_background else image_features
            
            # Normalize features before computing similarity
            features = features / features.norm(dim=-1, keepdim=True)
            class_avg_feature = class_avg_feature / class_avg_feature.norm(dim=-1, keepdim=True)
            
            # Compute similarity with temperature scaling
            similarity = (class_avg_feature @ features.T)
            
            if self.dataset == "thumos":
                similarity = self.moving_average(
                    similarity.squeeze(), self.kernel_size
                ).unsqueeze(0)
            
            pindices, nindices = self.get_indices(similarity)
            image_features_p, image_features_n = image_features[pindices], image_features[nindices]
            
            # Normalize features
            image_features_p = image_features_p / image_features_p.norm(dim=-1, keepdim=True)
            image_features_n = image_features_n / image_features_n.norm(dim=-1, keepdim=True)
            
            # Compute similarities with temperature scaling
            similarity_p = (class_avg_feature @ image_features_p.T) / self.temperature
            similarity_n = (class_avg_feature @ image_features_n.T) / self.temperature

            # Debug prints
            #print(f"Step {step} - Before optimization:")
            #print(f"Max similarity_p: {similarity_p.max().item():.4f}, Min similarity_p: {similarity_p.min().item():.4f}")
            #print(f"Max similarity_n: {similarity_n.max().item():.4f}, Min similarity_n: {similarity_n.min().item():.4f}")

            similarity_p = similarity_p.unsqueeze(0)
            similarity_n = similarity_n.unsqueeze(0)
            
            similarity = torch.cat([similarity_p.squeeze(), similarity_n.squeeze()], dim=0)
            gt = torch.cat(
                [torch.ones(similarity_p.shape[1]), torch.zeros(similarity_n.shape[1])],
                dim=0
            ).to(similarity.device)

            class_avg_feature = class_avg_feature.unsqueeze(0)
            
            if self.ltype in ["BYOL", "BCE"]:
                tta_loss = self.tta_loss(similarity, gt)
                
            elif self.ltype == "BYOLfeat":
                tta_loss = self.tta_loss(similarity, gt) + self.tta_loss(
                    image_features_p,
                    class_avg_feature.repeat_interleave(image_features_p.shape[0], dim=0),
                )
            elif self.ltype == "HY":
                #pindices, nindices = self.get_indices(similarity)
        
                # Ensure similarity is properly shaped before indexing
                similarity = similarity.view(-1)
                tta_loss = self.hybrid_loss(
                    pos_sim=similarity_p,
                    neg_sim=similarity_n,
                    features=image_features,
                    prototypes=self.avg_features.to(image_features.device)
                )

            else:
                raise ValueError(f"Not implemented loss type: {self.ltype}")

            print(f"Step {step} - TTA Loss: {tta_loss.item():.4f}")
            tta_loss.backward(retain_graph=True)
            
            # Add gradient clipping
            if image_features_pre.grad is not None:
                grad_norm = image_features_pre.grad.norm().item()
                #print(f"Step {step} - Gradient norm: {grad_norm:.4f}")
                torch.nn.utils.clip_grad_norm_([image_features_pre], max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()


        with torch.no_grad():
            # Use average feature for final prediction
            class_avg_feature = self.avg_features[indexes.item()].to(image_features.device)
            
            if self.remove_background:
                image_features = image_features - self.background_embedding
            
            image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (class_avg_feature @ image_features_norm.T)
            
            if self.dataset == "thumos":
                similarity = self.moving_average(similarity.squeeze(), self.kernel_size)
            if self.normalize:
                similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min())
            
            similarity = similarity.squeeze()
            segments = self.select_segments(similarity)
            pred_mask = torch.zeros(image_features.shape[0]).to(image_features.device)
            gt_mask = torch.zeros(image_features.shape[0]).to(image_features.device)

            if segments:
                image_features = [
                    torch.mean(image_features[seg[0] : seg[1]], dim=0)
                    for seg in segments
                ]
                image_features = torch.stack(image_features)
                pred, scores = self.compute_score(image_features, self.avg_features)
                
                for seg in segments:
                    pred_mask[seg[0] : seg[1]] = 1
                for anno in segments_gt:
                    gt_mask[anno[0] : anno[1]] = 1
                    
                output = [
                    {
                        "label": indexes.item(),
                        "score": scores[i],
                        "segment": segments[i],
                    }
                    for i in range(len(segments))
                ]
            else:
                output = [
                    {
                        "label": -1,
                        "score": 0,
                        "segment": [],
                    }
                ]

        if self.image_projection:
            self.model.visual.load_state_dict(before_optimization_parameters_image_encoder)
            
        if self.visualize:
            sim_plot = self.plot_visualize(
                video_name, similarity, indexes, segments_gt, segments, unique_labels
            )
        else:
            sim_plot = None
            
        return (
            video_name,
            output,
            pred_mask,
            gt_mask,
            unique_labels,
            sim_plot,
        )

    # Other helper methods remain unchanged
    def moving_average(self, data, window_size):
        padding_size = window_size
        padded_data = torch.cat(
            [
                torch.ones(padding_size).to(data.device) * data[0],
                data,
                torch.ones(padding_size).to(data.device) * data[-1],
            ]
        )
        kernel = (torch.ones(window_size) / window_size).to(data.device)
        smoothed_data = F.conv1d(padded_data.view(1, 1, -1), kernel.view(1, 1, -1))
        smoothed_data = smoothed_data.view(-1)[padding_size // 2 + 1 : -padding_size // 2]
        return smoothed_data

    def get_video_fps(self, video_name):
        video_extensions = [".mp4", ".mkv", ".webm"]
        for ext in video_extensions:
            video_path = os.path.join(self.video_dir, video_name + ext)
            if os.path.exists(video_path):
                fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
                break
            else:
                print(f"Video {video_name} not found in {video_path}")
                continue
        return fps

    def get_segments_gt(self, video_name, fps):
        segments_gt = [
            anno["segment"]
            for anno in self.annotations[video_name]["annotations"]
            if anno["label"] in self.cls_names
        ]
        segments_gt = [
            [int(float(seg[0]) * fps), int(float(seg[1]) * fps)] for seg in segments_gt
        ]
        label_gt = [
            anno["label"]
            for anno in self.annotations[video_name]["annotations"]
            if anno["label"] in self.cls_names
        ]
        unique_labels = set(label_gt)
        return segments_gt, unique_labels
    
    def get_indices(self, signal):
        if (100 * self.n) >= signal.shape[1]:
            pindices = torch.arange(signal.shape[1]).to("cuda")
            nindices = torch.arange(signal.shape[1]).to("cuda")
        else:
            pindices = torch.topk(signal, (100 * self.n) % signal.shape[1])[1]
            nindices = torch.topk(-signal, (100 * self.n) % signal.shape[1])[1]
        pindices = pindices.squeeze().sort()[0]
        nindices = nindices.squeeze().sort()[0]
        if pindices.shape[0] < self.n:
            pindices = pindices.repeat_interleave(self.n // pindices.shape[0] + 1)
            pindices = pindices[: self.n]
        if nindices.shape[0] < self.n:
            nindices = nindices.repeat_interleave(self.n // nindices.shape[0] + 1)
            nindices = nindices[: self.n]
        pindices = pindices[:: (len(pindices) - 1) // (self.n - 1)][: self.n]
        nindices = nindices[:: (len(nindices) - 1) // (self.n - 1)][: self.n]
        pindices = torch.clamp(
            pindices
            + torch.randint(-self.randper, self.randper, (self.n,)).to(signal.device),
            0,
            signal.shape[1] - 1,
        )
        nindices = torch.clamp(
            nindices
            + torch.randint(-self.randper, self.randper, (self.n,)).to(signal.device),
            0,
            signal.shape[1] - 1,
        )
        return pindices, nindices

    def select_segments(self, similarity):
        #print(f"similarity: {similarity}")
        if self.dataset == "thumos":
            mask = similarity > similarity.mean()
        elif self.dataset == "anet":
            mask = similarity > self.m
        else:
            raise ValueError(f"Not implemented dataset: {self.dataset}")
        
        selected = torch.nonzero(mask).squeeze()
        segments = []
        if selected.numel() and selected.dim() > 0:
            interval_start = selected[0]
            for i in range(1, len(selected)):
                if selected[i] <= selected[i - 1] + self.stride:
                    continue
                else:
                    interval_end = selected[i - 1]
                    if interval_start != interval_end:
                        segments.append([interval_start.item(), interval_end.item()])
                    interval_start = selected[i]

            if interval_start != selected[-1]:
                segments.append([interval_start.item(), selected[-1].item()])

        return segments

    def plot_visualize(self, video_name, similarity, indexes, segments_gt, segment, unique_labels):
        fig = plt.figure(figsize=(25, 20))
        plt.scatter(
            torch.arange(similarity.shape[0]),
            similarity.detach().cpu().numpy(),
            c="darkblue",
            s=1,
            alpha=0.5,
        )
        plt.title(video_name)
        plt.text(
            0.7,
            0.9,
            f"{self.inverted_cls.get(indexes.item(), None)}",
            fontsize=20,
            transform=plt.gcf().transFigure,
            c="red",
        )
        for i, label in enumerate(unique_labels):
            plt.text(
                0.05,
                0.9 - i * 0.05,
                label,
                fontsize=20,
                transform=plt.gcf().transFigure,
                c="green",
            )
        for i, seg in enumerate(segments_gt):
            plt.axvspan(
                seg[0],
                seg[1],
                color="green",
                alpha=0.2,
            )
        for i, seg in enumerate(segment):
            plt.axvspan(
                seg[0],
                seg[1],
                color="red",
                alpha=0.1,
            )
        return plt