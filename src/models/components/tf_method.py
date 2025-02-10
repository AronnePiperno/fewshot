import torch
from torch import nn
from torch.functional import F
import open_clip
import matplotlib.pyplot as plt
import json
import os
import cv2
import re
import importlib
import numpy as np
tokenize = open_clip.get_tokenizer("coca_ViT-L-14")

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

        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14", pretrained="mscoco_finetuned_laion2B-s13B-b90k"
        )
        self.model = self.model.float()
        print(f"Loaded COCA model")

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
            self.annotations_path = (
                "./data/activitynet_annotations/anet_anno_action.json"
            )
            self.video_dir = os.path.join(self.video_path, "ActivityNetVideos/videos/")
        else:
            raise ValueError(f"Not implemented dataset: {self.dataset}")

        self.dict_test = getattr(
            importlib.import_module("config.zero_shot"), dict_test_name, None
        )

        self.cls_names = self.dict_test
        self.num_classes = len(self.cls_names)
        self.inverted_cls = {v: k for k, v in self.cls_names.items()}
        #self.text_features = self.get_text_features()

        with open(self.annotations_path, "r") as f:
            self.annotations = json.load(f)

        self.avg_video_features = self.load_avg_features(self.avg_features_path)
        self.text_features = self.get_text_features()
        print(f"classes loaded: {self.cls_names}")




    def load_avg_features(self, path):
        """Load precomputed average features (.npy) only for classes in cls_names."""
        avg_features = {}
        if not os.path.exists(path):
            raise FileNotFoundError(f"Average features folder not found: {path}")
        
        feature_name = {}
        i = 0
        for class_name in self.cls_names:
            feature_name[class_name] = i
            class_folder_path = os.path.join(path, class_name)
            avg_file_path = os.path.join(class_folder_path, class_name + "_average.npy")
            
            if os.path.exists(avg_file_path):
                feature_array = np.load(avg_file_path)

                avg_features[class_name] = torch.tensor(feature_array, dtype=torch.float32)
            else:
                print(f"Warning: Average feature file not found for class {class_name}")

            i += 1
        
        print(f"Loaded {len(avg_features)} class-specific average features.")

        
        # Average over the second dimension to reduce [10, 402, 768] -> [10, 768]
        avg_features_tensor = torch.stack([feature.mean(dim=0) for feature in avg_features.values()])
        print(f"feature name extracted{feature_name}")
        return avg_features_tensor
    
    def get_text_features(self):
        prompts = []
        for c in self.cls_names:
            c = re.sub(r"([a-z])([A-Z])", r"\1 \2", c)
            prompts.append("a video of action" + " " + c)

        text = [tokenize(p) for p in prompts]
        text = torch.stack(text)
        text = text.squeeze()
        text = text.to(next(self.model.parameters()).device)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def compute_score_pseudo(self, x, y):
        #normalize
        x = x / x.norm(dim=-1, keepdim=True)
        y = y / y.norm(dim=-1, keepdim=True)

        # Set the temperature scaling factor
        temperature = 0.3
        #self.model.logit_scale = torch.nn.Parameter(torch.tensor([.1]))
        
    
        with torch.no_grad():
            # Calculate dot product between image features and average features
            dot_product = (x @ y.T)
            #print(f"Dot product: {dot_product}")

            # Apply temperature scaling to logits
            scores = dot_product / temperature

        # Get the predicted class
        pred = scores.argmax(dim=-1)
        return pred, scores


    def infer_pseudo_labels(self, image_features):
        """Infer pseudo-labels using class-specific average features."""
        if image_features is None or image_features.numel() == 0:
            raise ValueError("image_features is empty or None.")

        # Average the image features
        image_features_avg = image_features.mean(dim=0)
        self.background_embedding = image_features_avg.unsqueeze(0)
        #self.text_features = self.text_features.to(image_features.device)

        # Check if avg_features is properly loaded
        if self.fuse_features is None or len(self.fuse_features) == 0:
            raise ValueError("avg_features is empty. Ensure class-specific features are loaded correctly.")

        scores = []
        for avg_feature in self.fuse_features:
            if avg_feature is None or avg_feature.numel() == 0:
                print(f"[Warning] avg_feature is empty or None. Skipping.")
                continue

            # Move avg_feature to the same device as image_features_avg
            avg_feature = avg_feature.to(image_features_avg.device)
            
            # Ensure avg_feature matches the shape of image_features_avg

            #avg_feature = avg_feature.mean(dim=0)
            
            # Compute scores
            _, scores_avg = self.compute_score_pseudo(image_features_avg, avg_feature)
            if scores_avg is None or scores_avg.numel() == 0:
                print(f"[Warning] compute_score returned None or empty. Skipping.")
                continue
            scores.append(scores_avg)

        if len(scores) == 0:
            raise RuntimeError("Scores is empty. Check avg_features and compute_score outputs.")

        # Convert scores to a tensor
        scores = torch.stack(scores)


        # Select top-k scores
        _, index = torch.topk(scores, self.topk)

        print(f"Top-k scores: {index}")

        return index[0], scores

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
        smoothed_data = smoothed_data.view(-1)[
            padding_size // 2 + 1 : -padding_size // 2
        ]
        return smoothed_data

    def plot_visualize(
        self, video_name, similarity, indexes, segments_gt, segment, unique_labels
    ):

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

    def compute_score(self, model, x, y):
        #x = x / x.norm(dim=-1, keepdim=True)
        scores = x @ y.T
        pred = scores.argmax(dim=-1)
        return pred, scores


    def select_segments(self, similarity):
        if self.dataset == 'thumos':
            mask = similarity > similarity.mean()
        elif self.dataset == 'anet': 
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
    
    def get_video_fps(self, video_name):
        video_extensions = [".mp4", ".mkv", ".webm"]
        for ext in video_extensions:
            video_path = os.path.join(self.video_dir, video_name + ext)
            if os.path.exists(video_path):
                fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
                break
        return fps
    


    def forward(self, x):
        idx, video_name, image_features = x
        video_name = video_name[0]
        with torch.no_grad():
            image_features = image_features @ self.model.visual.proj
        image_features = image_features.squeeze(0)

        
        # vision-language fusion query
        text_features = self.text_features.to(image_features.device)
        video_features = self.avg_video_features.to(image_features.device)

        self.fuse_features = (text_features + video_features) / 2
        print(self.fuse_features.shape)


        indexes, _ = self.infer_pseudo_labels(image_features)
        class_label = self.inverted_cls[indexes.item()]

        pseudolabel_feature = self.fuse_features[indexes]


        

        if self.remove_background:
            image_features = image_features - self.background_embedding

        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)


        similarity = (
            pseudolabel_feature @ image_features_norm.T
        )

        if self.dataset == "thumos":
            similarity = self.moving_average(similarity, self.kernel_size).squeeze()
        if self.normalize:
            similarity = (similarity - similarity.min()) / (
                similarity.max() - similarity.min()
            )
            
        fps = self.get_video_fps(video_name)
    
        segments_gt = [
            anno["segment"] for anno in self.annotations[video_name]["annotations"]
        ]
        segments_gt = [
            [int(float(seg[0]) * fps), int(float(seg[1]) * fps)] for seg in segments_gt
        ]
        label_gt = [
            anno["label"] for anno in self.annotations[video_name]["annotations"]
        ]
        unique_labels = set(label_gt)

        segments = self.select_segments(similarity)

        pred_mask = torch.zeros(image_features.shape[0]).to(image_features.device)
        gt_mask = torch.zeros(image_features.shape[0]).to(image_features.device)

        if segments:
            image_features = [
                torch.mean(image_features[seg[0] : seg[1]], dim=0) for seg in segments
            ]
            image_features = torch.stack(image_features)
            pred, scores = self.compute_score(
                self.model,
                image_features,
                self.fuse_features.to(image_features.device),
            )




            
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
                for i in range(pred.shape[0])
            ]
        else:
            output = [
                {
                    "label": -1,
                    "score": 0,
                    "segment": [],
                }
            ]

        if self.visualize:
            self.plot_visualize(
                video_name, similarity, indexes, segments_gt, segments, unique_labels
            )
            return (
                video_name,
                output,
                pred_mask,
                gt_mask,
                unique_labels,
                plt,
            )
        else:
            return (
                video_name,
                output,
                pred_mask,
                gt_mask,
                unique_labels,
                None,
            )
