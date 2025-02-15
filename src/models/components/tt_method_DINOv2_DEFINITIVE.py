import torch
from torch import nn
from torch.functional import F
from src.models.components.loss import ByolLoss
import open_clip
import matplotlib.pyplot as plt
import json
import os
import cv2
import re
import copy
import importlib
import numpy as np
import glob
from transformers import Dinov2Model


tokenize = open_clip.get_tokenizer("coca_ViT-L-14")


class Fusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim * 2, 4),
            nn.Linear(4, 2),             
            nn.Softmax(dim=-1)
        )
        
    
    def forward(self, text_feats, video_feats, only_video=False):

        if only_video:
            print("Only video")
            return video_feats

        combined = torch.cat([text_feats, video_feats], dim=-1)

        weights = self.attn(combined)
        print(f"Attention weights: {weights}")

        return weights[:, 0:1] * text_feats + weights[:, 1:2] * video_feats
    

class VideoProjector(nn.Module):
    def __init__(self, dim=768, dropout=0.1):
        super().__init__()

        self.proj_matrix = nn.Parameter(torch.eye(dim) + 0.001 * torch.randn(dim, dim))
        

        self.transform = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout)
        )

    def set_proj_matrix(self, proj_matrix):
        self.proj_matrix = proj_matrix
    
    def get_proj_matrix(self):
        return self.proj_matrix

    def forward(self, x):
        x = x.to(self.proj_matrix.device)
        x = self.transform(x)
        return x @ self.proj_matrix

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
        # attribute for average features
        avg_features_path: str, 
    ):
        super(T3ALNet, self).__init__()

        # Initialize all the previous parameters as before
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
        self.ltype = ltype
        self.steps = steps
        self.refine_with_captions = refine_with_captions
        self.split = 50
        self.setting = setting
        self.dataset = dataset
        self.visualize = visualize
        self.kernel_size = kernel_size
        self.video_path = video_path
        # attribute for average features
        self.avg_features_path = avg_features_path
        self.topk = 3
        self.m = 0.7

        # Load the COCA model
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14", pretrained="mscoco_finetuned_laion2B-s13B-b90k"
        )
        self.model = self.model.float()
        #self.model.train()
        print(f"Loaded COCA model")

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
                if self.setting == 10
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
        self.cls_names = self.dict_test
        print(f"Loaded {self.cls_names} classes for zero-shot learning.")
        self.num_classes = len(self.cls_names)
        self.inverted_cls = {v: k for k, v in self.cls_names.items()}
        self.text_features = self.get_text_features(self.model)



        with open(self.annotations_path, "r") as f:
            self.annotations = json.load(f)

        if self.ltype == "BCE":
            self.tta_loss = torch.nn.BCEWithLogitsLoss()
        elif "BYOL" in self.ltype:
            self.tta_loss = ByolLoss()
        else:
            raise ValueError(f"Not implemented loss type: {self.ltype}")
        

        self.avg_video_features = self.load_avg_features(self.avg_features_path)



            
        

        self.original_features = self.avg_video_features.clone()
        
        # create video projection tensor as identity tensor

        #self.video_proj = nn.Parameter(torch.eye(768) + 0.001 * torch.randn(768, 768))
        #self.video_proj = nn.Parameter(self.video_proj)

        self.video_proj = VideoProjector(dim=768)
        self.visual_proj = VideoProjector(dim=768)

        self.fusion = Fusion(embed_dim=768)
        

        self.optim = torch.optim.AdamW(
            [
                #{"params": self.model.parameters()},
                {"params": self.model.text.text_projection},
                #{"params": self.model.visual.proj, "lr": 0.00001*0.001},
                {"params": self.visual_proj.parameters(), "lr": 1e-5},
                {"params": self.fusion.parameters()},
                {"params": self.video_proj.parameters(), "lr": 1e-5},
                {"params": self.model.logit_scale},
            ], 
            lr= 0.00001, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=self.steps)
        self.only_support_videos = True
        

        


    def load_avg_features(self, path):
        """Load precomputed average features (.npy) only for classes in cls_names."""
        avg_features = {}
        if not os.path.exists(path):
            raise FileNotFoundError(f"Average features folder not found: {path}")
        
        for class_name in self.cls_names:
            class_folder_path = os.path.join(path, class_name)
            avg_file_path = os.path.join(class_folder_path, class_name + "_average.npy")
            
            if os.path.exists(avg_file_path):
                feature_array = np.load(avg_file_path)

                avg_features[class_name] = torch.tensor(feature_array, dtype=torch.float32)
            else:
                print(f"Warning: Average feature file not found for class {class_name}")
        
        print(f"Loaded {len(avg_features)} class-specific average features.")


        avg_features_tensor = torch.stack([feature for feature in avg_features.values()])

        return avg_features_tensor
    
    def load_support_features(self):
        path = "./data/Thumos14/support_videos_features"
        support_features = np.load(os.path.join(path, "stacked_features.npy"))

        # cast to tensor
        support_features = torch.tensor(support_features, dtype=torch.float32).to("cuda")
        return support_features



    def infer_pseudo_labels(self, image_features, fuse_features):
        """Infer pseudo-labels using class-specific average features."""
        if image_features is None or image_features.numel() == 0:
            raise ValueError("image_features is empty or None.")

        # Average the image features
        image_features_avg = image_features.mean(dim=0)
        self.background_embedding = image_features_avg.unsqueeze(0)
        #self.text_features = self.text_features.to(image_features.device)

        # Check if avg_features is properly loaded
        if fuse_features is None or len(fuse_features) == 0:
            raise ValueError("avg_features is empty. Ensure class-specific features are loaded correctly.")

        scores = []
        for avg_feature in fuse_features:
            if avg_feature is None or avg_feature.numel() == 0:
                print(f"[Warning] avg_feature is empty or None. Skipping.")
                continue

            # Move avg_feature to the same device as image_features_avg
            avg_feature = avg_feature.to(image_features_avg.device)
            # Ensure avg_feature matches the shape of image_features_avg

            #avg_feature = avg_feature.mean(dim=0)
            
            # Compute scores
            #_, scores_avg = self.compute_score(image_features_avg, avg_feature)
            _, scores_avg = self.compute_score(avg_feature, image_features_avg)
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

        #print("Fused features norm:", fuse_features.norm(dim=-1).mean().item())  # Should be ~1.0
        #print("Similarity range:", scores.min().item(), scores.max().item())  # Should be reasonable (e.g., -5 to 10)

        return index[0], scores


    def get_text_features(self, model):
        prompts = []
        for c in self.cls_names:
            c = re.sub(r"([a-z])([A-Z])", r"\1 \2", c)
            prompts.append("a video of action" + " " + c)

        text = [tokenize(p) for p in prompts]
        text = torch.stack(text)
        text = text.squeeze()
        text = text.to(next(model.parameters()).device)
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def compute_score(self, x, y):
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
            else:
                print(f"Video {video_name} not found in {video_path}")
                continue
        return fps

    """def infer_pseudo_labels(self, image_features):
        image_features_avg = image_features.mean(dim=0)
        self.background_embedding = image_features_avg.unsqueeze(0)
        self.text_features = self.text_features.to(image_features.device)
        _, scores_avg = self.compute_score(
            image_features_avg.unsqueeze(0),
            self.text_features,
        )
        _, indexes = torch.topk(scores_avg, self.topk)
        return indexes[0][0], scores_avg"""

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
        return plt

    def compute_tta_embedding(self, class_label, device):
        class_label = re.sub(r"([a-z])([A-Z])", r"\1 \2", class_label)
        
        # if class label is Billiards, change the name in pool
        '''if class_label == "Billiards":
            class_label = "pool sport"
        elif class_label == "ThrowDiscus":
            class_label = "disc throw"
        elif class_label == "TennisSwing":
            class_label = "tennis forehand"'''            


        class_label = "a video of action" + " " + class_label
        text = tokenize(class_label).to(device)
        tta_emb = self.model.encode_text(text)
        tta_emb = tta_emb / tta_emb.norm(dim=-1, keepdim=True)
        return tta_emb



    

    def forward(self, x, optimizer):
        self.video_proj.requires_grad_(True)
        self.visual_proj.requires_grad_(True)
        #self.model.requires_grad_(False)
        #torch.nn.utils.clip_grad_norm_(self.video_proj, max_norm=1.0)

        idx, video_name, image_features_pre = x
        image_features_pre = copy.deepcopy(image_features_pre)
        video_name = video_name[0]
        fps = self.get_video_fps(video_name)

        if not self.image_projection:
            image_features = image_features_pre
            image_features = image_features.squeeze(0)
        else:
            image_features_pre.requires_grad = True
            with torch.no_grad():
                #image_features = image_features_pre @ self.model.visual.proj
                image_features = image_features_pre
                image_features = image_features.squeeze(0)

        if not self.only_support_videos:
            text_features = self.get_text_features(self.model).to(image_features.device)
            text_features =  F.normalize(text_features, dim=-1)
        else:
            text_features = torch.zeros(768).to(image_features.device)

        video_features = self.avg_video_features.to(image_features.device)
        #video_features.requires_grad = True


        #video_features =  self.video_proj(video_features)
        
        #self.fuse_features = self.fusion_weights[0] * text_features + self.fusion_weights[1] * video_features

        self.fuse_features = self.fusion(text_features, video_features, only_video=self.only_support_videos)


                
        indexes, scores_to_return = self.infer_pseudo_labels(image_features, self.fuse_features.to(image_features.device))
        class_label = self.inverted_cls[indexes.item()]
        print(f"Class label: {class_label}")

        segments_gt, unique_labels = self.get_segments_gt(video_name, fps)

        original_video_features = self.avg_video_features[indexes.item()].unsqueeze(0).to(image_features.device)

        before_optimization_video_projection = copy.deepcopy(self.video_proj.get_proj_matrix())
        before_optimization_visual_projection = copy.deepcopy(self.visual_proj.get_proj_matrix())

        for _ in range(self.steps):
            if self.image_projection:
                #image_features = (image_features_pre @ self.model.visual.proj).squeeze(0)
                image_features = self.visual_proj(image_features_pre).squeeze(0)
                before_optimization_parameters_image_encoder = copy.deepcopy(self.model.visual.state_dict())
                before_optimization_image_projection = copy.deepcopy(self.model.visual.proj)
                

            if self.text_projection:
                before_optimization_text_projection = copy.deepcopy(self.model.text.text_projection)
                before_optimization_parameters_text_encoder = copy.deepcopy(self.model.text.state_dict())
            else:
                before_optimization_parameters_text_encoder = copy.deepcopy(
                    self.model.text.state_dict()
                )
            before_optimization_logit_scale = copy.deepcopy(self.model.logit_scale)
            
            if not self.only_support_videos:
                #print("Not only support videos")
                text_features = self.compute_tta_embedding(class_label, image_features.device)
                text_features = F.normalize(text_features, dim=-1)
            else:
                pass
                #text_features = torch.zeros(768).to(image_features.device)
            
            features = image_features - self.background_embedding if self.remove_background else image_features





            #projected_video_features = video_features @ self.video_proj
            projected_video_features = self.video_proj(original_video_features)




            tta_emb = self.fusion(text_features, projected_video_features, only_video=self.only_support_videos)

            
            
            tta_emb = tta_emb.to(image_features.device)

            tta_emb = F.normalize(tta_emb, dim=-1)

            print(f"tta_emb shape {tta_emb.shape}")
            print(f"image_features shape {image_features.shape}")

            similarity = self.model.logit_scale.exp() * tta_emb @ features.T



            
            if self.dataset == "thumos":
                similarity = self.moving_average(
                    similarity.squeeze(), self.kernel_size
                ).unsqueeze(0)
            


            pindices, nindices = self.get_indices(similarity)
            image_features_p, image_features_n = image_features[pindices], image_features[nindices]


            image_features_p = image_features_p / image_features_p.norm(
                dim=-1, keepdim=True
            )
            image_features_n = image_features_n / image_features_n.norm(
                dim=-1, keepdim=True
            )

            #self.update_support_features(image_features_p, indexes.item())
            #video_features = self.feature_memory['video_features'][indexes.item()].to(image_features.device)

            similarity_p = (
                self.model.logit_scale.exp() * tta_emb @ image_features_p.T
            )
            similarity_n = (
                self.model.logit_scale.exp() * tta_emb @ image_features_n.T
            )
            similarity = torch.cat(
                [similarity_p.squeeze(), similarity_n.squeeze()], dim=0
            )


            gt = torch.cat(
                [
                    torch.ones(similarity_p.shape[1]),
                    torch.zeros(similarity_n.shape[1]),
                ],
                dim=0,
            ).to(similarity.device)
            


            
            #proto_loss = self.prototypical_loss(tta_emb, self.avg_video_features, indexes.item())
            #tta_emb = tta_emb.unsqueeze(0)
            if self.ltype in ["BYOL", "BCE"]:
                tta_loss = self.tta_loss(similarity, gt)
            elif self.ltype == "BYOLfeat":
                tta_loss = self.tta_loss(similarity, gt) + self.tta_loss(
                    image_features_p,
                    tta_emb.repeat_interleave(image_features_p.shape[0], dim=0),
                )
                #print(f"step {_} - TTA Loss: {tta_loss.item():.4f}")
            else:
                raise ValueError(f"Not implemented loss type: {self.ltype}")
            
           # negatives = self.generate_negatives(video_features, indexes.item(), image_features.device)
            #contrast_loss = self.compute_contrastive_loss(tta_emb, video_features, negatives)

            #tta_loss = self.tta_loss(similarity, gt) + self.lambda_contrast * contrast_loss
            #tta_loss = contrast_loss


            # print magnitudes of self.video_proj gradients

            


            #tta_loss = tta_loss +  proto_loss
            
            if _ % 10 ==  0:
                print(f"Step {_} - TTA Loss: {tta_loss.item():.4f}")


            
            tta_loss.backward(retain_graph=True)


            self.optim.step()



            self.optim.zero_grad()
            self.scheduler.step()

        # check if the parameters are update
        #self.original_features = self.original_features.to(image_features.device)



        #assert not torch.equal(
        #    before_optimization_video_projection,
        #    copy.deepcopy(self.video_proj.get_proj_matrix()),
        #), f"Parameter video_features has not been updated."

        if torch.equal( before_optimization_video_projection, copy.deepcopy(self.video_proj.get_proj_matrix())):
            print(f"Before optimization video projection matrix: {before_optimization_video_projection}")
            print(f"After optimization video projection matrix: {self.video_proj.get_proj_matrix()}")
            print(f"Video projection matrix has not been updated.")

        #assert not torch.equal(
        #    before_optimization_visual_projection,
        #    copy.deepcopy(self.visual_proj.get_proj_matrix()),
        #), f"Parameter visual_proj has not been updated."



        if not self.only_support_videos:
            if self.text_projection:
                assert not torch.equal(
                    before_optimization_text_projection,
                    copy.deepcopy(self.model.text.text_projection),
                ), f"Parameter text_projection has not been updated."

        #if self.image_projection :
        #    assert not torch.equal(
        #        before_optimization_image_projection,
        #        copy.deepcopy(self.model.visual.proj),
        #    ), f"Parameter has not been updated."

        with torch.no_grad():
            if not self.only_support_videos:
                text_features = self.compute_tta_embedding(class_label, image_features.device)
                text_features = F.normalize(text_features, dim=-1)
            else:
                pass
                #text_features = torch.zeros(768).to(image_features.device)
            
            video_features = self.avg_video_features[indexes.item()].unsqueeze(0)
            video_features = video_features.to('cuda')
            video_features = self.video_proj(video_features)
            video_features = F.normalize(video_features, dim=-1)
           

            tta_emb = self.fusion(text_features, video_features, only_video=self.only_support_videos)

            #if self.remove_background:
            #    image_features = image_features - self.background_embedding
            
            image_features_norm = image_features / image_features.norm(
                dim=-1, keepdim=True
            )

            tta_emb = tta_emb.to(image_features.device)
            tta_emb = F.normalize(tta_emb, dim=-1)

            similarity = tta_emb @ image_features_norm.T

            
            if self.dataset == "thumos":
                similarity = self.moving_average(similarity.squeeze(), self.kernel_size)
            if self.normalize:
                similarity = (similarity - similarity.min()) / (
                    similarity.max() - similarity.min()
                )
            similarity = similarity.squeeze()

            segments = self.select_segments(similarity)
            pred_mask = torch.zeros(image_features.shape[0]).to(image_features.device)
            gt_mask = torch.zeros(image_features.shape[0]).to(image_features.device)
            after_optimization_text_encoder = copy.deepcopy(
                self.model.text.state_dict()
            )
            after_optimization_logit_scale = copy.deepcopy(self.model.logit_scale)
            
            
            if self.refine_with_captions and len(segments) > 1:
                self.model.logit_scale = before_optimization_logit_scale
                self.model.text.load_state_dict(
                    before_optimization_parameters_text_encoder
                )

                with open(f"./data/Thumos14/captions/{video_name}.txt", "r", encoding='utf-8') as f:
                    captions = f.readlines()
                captions = [
                    (int(c.split("-")[0].split(".")[0]) * 3, c.split("-")[1])
                    for c in captions
                ]
                captions_per_segment = [[] for _ in range(len(segments))]
                image_features_per_segment = [[] for _ in range(len(segments))]
                for i, seg in enumerate(segments):
                    image_features_per_segment[i] = image_features[seg[0] : seg[1]]
                    for cap in captions:
                        if cap[0] >= seg[0] and cap[0] <= seg[1]:
                            captions_per_segment[i].append((cap[1]))
                captions_per_segment = [
                    [tokenize(p) for p in cap] for cap in captions_per_segment
                ]
                segments = [
                    seg
                    for seg, cap in zip(segments, captions_per_segment)
                    if len(cap) > 0
                ]
                captions_per_segment = [
                    cap for cap in captions_per_segment if len(cap) > 0
                ]
                captions_per_segment = [
                    torch.stack(cap) for cap in captions_per_segment
                ]
                captions_per_segment = [cap.squeeze() for cap in captions_per_segment]
                captions_per_segment = [
                    cap.to(image_features.device) for cap in captions_per_segment
                ]
                captions_per_segment = [
                    cap.unsqueeze(0) if len(cap.shape) == 1 else cap
                    for cap in captions_per_segment
                ]
                captions_per_segment = [
                    self.model.encode_text(cap) for cap in captions_per_segment
                ]
                captions_per_segment = [cap.mean(dim=0) for cap in captions_per_segment]
                captions_per_segment = [
                    cap / cap.norm(dim=-1, keepdim=True) for cap in captions_per_segment
                ]
                similarity_with_other_captions = []
                for cap in captions_per_segment:
                    similarity_with_other_captions.append(
                        cap @ torch.stack(captions_per_segment).T
                    )
                segments = [
                    seg
                    for seg, sim in zip(segments, similarity_with_other_captions)
                    if torch.sum(sim > self.p) > len(segments) // 2
                ]
                self.model.logit_scale = after_optimization_logit_scale
                self.model.text.load_state_dict(after_optimization_text_encoder)

            if segments:
                image_features = [
                    torch.mean(image_features[seg[0] : seg[1]], dim=0)
                    for seg in segments
                ]

                if not self.only_support_videos:
                    text_features = self.get_text_features(self.model).to(image_features[0].device)
                    text_features =  F.normalize(text_features, dim=-1)
                else:
                    text_features = torch.zeros(768).to(image_features[0].device)

                video_features = self.avg_video_features
                video_features = video_features.to(image_features[0].device)
                video_features = F.normalize(video_features, dim=-1)
                video_features = self.video_proj(video_features)
                


                fuse_features = self.fusion(text_features, video_features, only_video=self.only_support_videos)

                image_features = torch.stack(image_features)
                #text_features = self.get_text_features(self.model).to(image_features.device)

                pred, scores = self.compute_score(
                    image_features,
                    fuse_features.to(image_features.device),
                )
                
                for seg in segments:
                    pred_mask[seg[0] : seg[1]] = 1
                for anno in segments_gt:
                    gt_mask[anno[0] : anno[1]] = 1

                

                #print(f"scores: {scores}")
                output = [
                    {
                        "label": indexes.item(),
                        "score": scores[i],
                        "segment": segments[i],
                    }
                    for i in range((len(segments))) 
                ]

                if len(output) == 0:
                    output = [
                        {
                            "label": -1,
                            "score": 0,
                            "segment": [],
                        }
                    ]

            else:
                output = [
                    {
                        "label": -1,
                        "score": 0,
                        "segment": [],
                    }
                ]

        #self.restore_original_features()

        print(f"Output: {output}")

        self.model.text.load_state_dict(before_optimization_parameters_text_encoder)
        self.model.logit_scale = before_optimization_logit_scale
        if self.image_projection:
            self.model.visual.load_state_dict(
                before_optimization_parameters_image_encoder
            )

        # reinitialize the video projection matrix

        with torch.no_grad():
            init_matrix = torch.eye(768).to(self.video_proj.proj_matrix.device) + 0.001 * torch.randn(768, 768).to(self.video_proj.proj_matrix.device)
            self.video_proj.proj_matrix.data = init_matrix
            # Also reinitialize LayerNorm and Dropout if needed
            for layer in self.video_proj.transform.children():
                if isinstance(layer, nn.LayerNorm):
                    layer.reset_parameters()
                elif isinstance(layer, nn.Dropout):
                    layer.p = 0.1

        with torch.no_grad():
            init_matrix = torch.eye(768).to(self.visual_proj.proj_matrix.device) + 0.001 * torch.randn(768, 768).to(self.visual_proj.proj_matrix.device)
            self.visual_proj.proj_matrix.data = init_matrix
            # Also reinitialize LayerNorm and Dropout if needed
            for layer in self.visual_proj.transform.children():
                if isinstance(layer, nn.LayerNorm):
                    layer.reset_parameters()
                elif isinstance(layer, nn.Dropout):
                    layer.p = 0.1

        
            

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
            None,
        )