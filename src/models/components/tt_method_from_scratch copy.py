import torch
import torch.nn as nn
import open_clip
import os
import numpy as np
import importlib
import cv2
import json

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
        self.ltype = "TTA"
        self.steps = 50
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
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14", pretrained="mscoco_finetuned_laion2B-s13B-b90k"
        )

        

                # Dataset-specific configuration
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
            self.video_dir = os.path.join(self.video_path, "ActivityNet/videos/")
        else:
            raise ValueError(f"Not implemented dataset: {self.dataset}")
        
        self.dict_test = getattr(
            importlib.import_module("config.zero_shot"), dict_test_name, None
        )

        with open(self.annotations_path, "r") as f:
            self.annotations = json.load(f)
            
        self.cls_names = self.dict_test
        print(f"Loaded {self.cls_names} classes for zero-shot learning.")
        self.num_classes = len(self.cls_names)
        self.inverted_cls = {v: k for k, v in self.cls_names.items()}

        self.support_videos_features = self.import_support_videos_features()
        print('Support videos features')
        print(self.support_videos_features.keys())
        #print dimensions of support videos features
        for k, v in self.support_videos_features.items():
            print(k, v.shape)
        
        self.support_videos_features_tensor = torch.cat(list(self.support_videos_features.values()), dim=0).to(self.DEVICE)
        self.support_videos_features_tensor = self.support_videos_features_tensor / torch.norm(self.support_videos_features_tensor, dim=1, keepdim=True)
        self.support_videos_features_tensor = self.support_videos_features_tensor.float()
        


    def import_support_videos_features(self):
        features = {}
        
        for cls in self.cls_names:
            class_folder = os.path.join(self.avg_features_path, cls)

            avg_file_path = os.path.join(class_folder, cls + '_average.npy')
            avg_features = np.load(avg_file_path)

            features[cls] = avg_features.mean(axis=0)

            # create tensor
            features[cls] = torch.tensor(features[cls]).unsqueeze(0)

        return features
        
    def calculate_similarity(self, x, y):
        # normalize the input
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = y / torch.norm(y, dim=-1, keepdim=True)

        return torch.matmul(x, y.t())
    
    def select_segments(self, similarity):
        
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
    
    def get_video_fps(self, video_name):
        video_extensions = [".mp4", ".mkv", ".webm"]
        for ext in video_extensions:
            video_path = os.path.join(self.video_dir, video_name + ext)

            if os.path.exists(video_path):
                fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
                break
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

    


    def forward(self, x, optimizer):
        idx, video_name, video_embeddings = x
        video_name = video_name[0]
        video_embeddings = video_embeddings.squeeze(0).to(self.DEVICE)

        video_embeddings_mean = video_embeddings.mean(dim=0).unsqueeze(0)

        video_embeddings_mean = video_embeddings_mean / torch.norm(video_embeddings_mean, dim=-1, keepdim=True)

        # calculate the similarity between all the support videos and the query video
        similarities = self.calculate_similarity(video_embeddings_mean, self.support_videos_features_tensor)

        #check the highest similarity
        max_sim, max_idx = torch.max(similarities, dim=1)

        print(f"max idx: {max_idx}")


        #check segments giving the tensor in max_idx
        similarities = self.calculate_similarity(video_embeddings, self.support_videos_features_tensor[max_idx].squeeze(0))

        segments = self.select_segments(similarities)

        fps = self.get_video_fps(video_name)
        print(f"shape of video_embeddings: {video_embeddings.shape}")
        pred_mask = torch.zeros(video_embeddings.shape[0]).to(self.DEVICE)
        gt_mask = torch.zeros(video_embeddings.shape[0]).to(self.DEVICE)
        segments_gt, unique_labels = self.get_segments_gt(video_name, fps)

        if segments:
                video_embeddings = [
                    torch.mean(video_embeddings[seg[0] : seg[1]], dim=0)
                    for seg in segments
                ]
                
                video_embeddings = torch.stack(video_embeddings).to(self.DEVICE)
                video_embeddings = video_embeddings / torch.norm(video_embeddings, dim=-1, keepdim=True)
                scores = self.calculate_similarity(video_embeddings, self.support_videos_features_tensor.squeeze(0))

                print(f"shape of score: {scores.shape}")

                for seg in segments:
                    pred_mask[seg[0] : seg[1]] = 1
                for anno in segments_gt:
                    gt_mask[anno[0] : anno[1]] = 1

                # cast max_idx to int
                

                output = [
                    {
                        "label": max_idx.item(),
                        "score": scores[i],
                        "segment": segments[i],
                    }
                    for i in range((len(segments)))
                ]
                #print(f"output: {output}")
        else:
            output = [
                {
                    "label": -1,
                    "score": 0,
                    "segment": [],
                }
            ]

        return (
            video_name,
            output,
            pred_mask,
            gt_mask,
            unique_labels,
            None,
        )

        
