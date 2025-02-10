# FIRST STAGE: IMAGE
# SECOND STAGE: TEXT

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
import decord
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

tokenize = open_clip.get_tokenizer("coca_ViT-L-14")


class TTALoss(nn.Module):
    def __init__(self, temperature=0.1, contra_weight=0.5, consist_weight=0.5):
        """
        Custom loss for Test Time Adaptation combining contrastive and consistency losses.
        
        Args:
            temperature (float): Temperature parameter for scaling similarities
            contra_weight (float): Weight for contrastive loss component
            consist_weight (float): Weight for consistency loss component
        """
        super(TTALoss, self).__init__()
        self.temperature = temperature
        self.contra_weight = contra_weight
        self.consist_weight = consist_weight
        
    def forward(self, query_features, support_features, positive_indices=None):
        """
        Compute the TTA loss between query and support features.
        
        Args:
            query_features (torch.Tensor): Features from the query/test image [N, D]
            support_features (torch.Tensor): Features from support/reference images [M, D]
            positive_indices (torch.Tensor, optional): Indices of positive pairs [N]
            
        Returns:
            torch.Tensor: Combined TTA loss
        """
        # Normalize features
        query_features = F.normalize(query_features, dim=-1)
        support_features = F.normalize(support_features, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(query_features, support_features.T) / self.temperature
        
        # Contrastive loss component
        contra_loss = self._compute_contrastive_loss(sim_matrix, positive_indices)
        
        # Feature consistency loss component
        consist_loss = self._compute_consistency_loss(query_features, support_features)
        
        # Combine losses
        total_loss = self.contra_weight * contra_loss + self.consist_weight * consist_loss
        
        return total_loss
    
    def _compute_contrastive_loss(self, sim_matrix, positive_indices=None):
        """
        Compute supervised contrastive loss if positive_indices provided,
        otherwise compute unsupervised contrastive loss.
        """
        if positive_indices is not None:
            # Supervised contrastive loss
            exp_sim = torch.exp(sim_matrix)
            positive_mask = torch.zeros_like(sim_matrix)
            positive_mask.scatter_(1, positive_indices.unsqueeze(1), 1)
            
            numerator = torch.sum(sim_matrix * positive_mask, dim=1)
            denominator = torch.sum(exp_sim, dim=1)
            losses = -numerator + torch.log(denominator)
            
            return torch.mean(losses)
        else:
            # Unsupervised contrastive loss (similar to SimCLR)
            exp_sim = torch.exp(sim_matrix)
            soft_max_prob = exp_sim / exp_sim.sum(dim=1, keepdim=True)
            losses = -torch.log(soft_max_prob.diagonal())
            
            return torch.mean(losses)
    
    def _compute_consistency_loss(self, query_features, support_features):
        """
        Compute feature consistency loss to maintain stable predictions.
        """
        # Compute mean feature vectors
        query_mean = torch.mean(query_features, dim=0)
        support_mean = torch.mean(support_features, dim=0)
        
        # L2 distance between normalized mean features
        consistency_loss = torch.norm(F.normalize(query_mean) - F.normalize(support_mean), p=2)
        
        return consistency_loss

    def compute_confidence_threshold(self, sim_matrix, percentile=90):
        """
        Compute adaptive confidence threshold based on similarity distribution.
        """
        flat_sim = sim_matrix.view(-1)
        threshold = torch.quantile(flat_sim, percentile/100.0)
        return threshold

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
        self.n = 2
        self.normalize = normalize
        self.text_projection = False
        self.text_encoder = text_encoder
        self.image_projection = image_projection
        self.logit_scale = logit_scale
        self.remove_background = remove_background
        self.ltype = "BCE"
        self.steps = 1
        self.refine_with_captions = False
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
        self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14", pretrained="mscoco_finetuned_laion2B-s13B-b90k"
        )
        self.model = self.model.float()
        print(f"Loaded COCA model")

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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Rest of the initialization remains the same
        self.dict_test = getattr(
            importlib.import_module("config.zero_shot"), dict_test_name, None
        )
        self.cls_names = self.dict_test
        print(f"Loaded {self.cls_names} classes for zero-shot learning.")
        self.num_classes = len(self.cls_names)
        self.inverted_cls = {v: k for k, v in self.cls_names.items()}
        #self.text_features = self.get_text_features(self.model)
        
        # Load average features
        self.avg_features = self.load_avg_features(self.avg_features_path)


        with open(self.annotations_path, "r") as f:
            self.annotations = json.load(f)

        if self.ltype == "BCE":
            self.tta_loss = torch.nn.BCEWithLogitsLoss()
        elif "BYOL" in self.ltype:
            self.tta_loss = ByolLoss()
        elif self.ltype == "TTA":
            self.tta_loss = TTALoss()
        else:
            raise ValueError(f"Not implemented loss type: {self.ltype}")
        
        


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

        
        # Average over the second dimension to reduce [10, 402, 768] -> [10, 768]
        avg_features_tensor = torch.stack([feature.mean(dim=0) for feature in avg_features.values()])

        return avg_features_tensor

    def infer_pseudo_labels(self, image_features):
        """Infer pseudo-labels using class-specific average features."""
        if image_features is None or image_features.numel() == 0:
            raise ValueError("image_features is empty or None.")

        # Average the image features
        image_features_avg = image_features.mean(dim=0)
        self.background_embedding = image_features_avg.unsqueeze(0)
        #self.text_features = self.text_features.to(image_features.device)

        # Check if avg_features is properly loaded
        if self.avg_features is None or len(self.avg_features) == 0:
            raise ValueError("avg_features is empty. Ensure class-specific features are loaded correctly.")

        scores = []
        for avg_feature in self.avg_features:
            if avg_feature is None or avg_feature.numel() == 0:
                print(f"[Warning] avg_feature is empty or None. Skipping.")
                continue

            # Move avg_feature to the same device as image_features_avg
            avg_feature = avg_feature.to(image_features_avg.device)
            
            # Ensure avg_feature matches the shape of image_features_avg

            #avg_feature = avg_feature.mean(dim=0)
            
            # Compute scores
            _, scores_avg = self.compute_score(image_features_avg, avg_feature)
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

        return index[0], scores


    def get_support_image_features(self, model):
        """Extract image features from the model."""
        image_features = {}
        SUPPORT_VIDEOS_PATH = "./data/Thumos14/support_videos/"
        for class_name in self.cls_names:
            class_folder_path = os.path.join(SUPPORT_VIDEOS_PATH, class_name)
            for video in os.listdir(class_folder_path):
                video_path = os.path.join(class_folder_path, video)
                video_features = self.extract_video_features(video_path)

                # cast to tensor
                video_features = torch.tensor(video_features, dtype=torch.float32)

                # average the 0th dimension
                video_features_avg = video_features.mean(dim=0)
                image_features[class_name] = video_features_avg
                
                


        #image_features = [torch.tensor(f) if isinstance(f, np.ndarray) else f for f in image_features]
        
        image_features = torch.stack([torch.tensor(f, dtype=torch.float32) if isinstance(f, np.ndarray) or isinstance(f, np.generic) else f for f in image_features.values()])

        
        return image_features

    
    def compute_score(self, x, y):


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
        print(f"Signal shape: {signal.shape}")
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

    
    def extract_video_features(self, video_path) -> np.ndarray:
        """Extract frame features from a video file"""
        FRAME_STRIDE = 15
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        BATCH_SIZE = 64

        try:
            vr = decord.VideoReader(video_path, num_threads=4)
            frame_indices = range(0, len(vr), FRAME_STRIDE)
            
            if not frame_indices:
                return np.empty((0, self.model.visual.output_dim), dtype=np.float16)
            
            frames = vr.get_batch(frame_indices).asnumpy()
            preprocessed = torch.stack([
                self.preprocess(Image.fromarray(frame)) for frame in frames
            ]).to(DEVICE)

            features = []
            with torch.no_grad(), torch.cuda.amp.autocast():
                for batch in torch.split(preprocessed, BATCH_SIZE):
                    features.append(self.model.encode_image(batch).cpu())
            return torch.cat(features).numpy().astype(np.float16)
        
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            return np.empty((0, self.model.visual.output_dim), dtype=np.float16)

    def compute_tta_embedding(self, class_label, device):
        if self.dataset == "thumos":
            support_videos_path = "./data/Thumos14/support_videos/"
        elif self.dataset == "anet":
            support_videos_path = "./data/ActivityNet/support_videos/"
        else:
            raise ValueError(f"Not implemented dataset: {self.dataset}")
        

        for video in os.listdir(os.path.join(support_videos_path, class_label)):
            video_path = os.path.join(support_videos_path, class_label,video)
            video_features = self.extract_video_features(video_path)

        tta_emb = torch.tensor(video_features.mean(axis=0)).to(device)

        return tta_emb.float()
        

    def visualize_embeddings(self, features, labels, title="t-SNE Visualization of Features"):
        """
        Visualize embeddings using t-SNE with different colors for TTA embeddings and features.
        
        Args:
            features (torch.Tensor): Combined tensor of TTA embeddings and features
            labels (list): Labels indicating whether each point is a TTA embedding or feature
            title (str): Title for the plot
        """

        
        # Ensure features are on CPU and converted to numpy
        if torch.is_tensor(features):
            features = features.detach().cpu().numpy()
        
        # Handle labels
        if isinstance(labels, (list, tuple)):
            # If labels is already a list, convert to numpy array

            #convert list of tensors to list of numpy arrays
            labels = [label.detach().cpu().numpy() if torch.is_tensor(label) else label for label in labels]

            labels = np.array(labels)
        elif torch.is_tensor(labels):
            # If labels is a tensor, move to CPU and convert to numpy
            labels = labels.detach().cpu().numpy()
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
        features_tsne = tsne.fit_transform(features)
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Create a scatter plot with different colors for TTA embeddings and features
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap('tab10', 12)
        print(f"Unique labels: {unique_labels}")
        for i, label in enumerate(unique_labels):
            print(f"Label: {label}")

            mask = labels == label
            plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                        color=colors(i), label=f'{label}')
        
        plt.title(title)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add a text box with information about the number of points
        info_text = f'Total points: {len(features)}\n'
        for i, label in enumerate(unique_labels):
            info_text += f'{label}: {np.sum(labels == label)}\n'
        
        plt.text(0.02, 0.98, info_text,
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        plt.show()


    def check_for_collapse(self, tta_emb, features, labels=None):
        """
        Comprehensive analysis of feature embeddings and TTA embeddings to check for collapse.
        
        Args:
            tta_emb (torch.Tensor): TTA embeddings
            features (torch.Tensor): Feature embeddings
            labels (array-like, optional): Labels for visualization
        """

        print(f"TTA Embeddings: {tta_emb.shape}")
        print(f"Features: {features.shape}")
        
        # Combine TTA embeddings and features
        combined_features = torch.cat([tta_emb, features], dim=0)
        

        # Calculate and print similarity statistics
        with torch.no_grad():
            # Normalize features
            tta_emb_norm = tta_emb / tta_emb.norm(dim=1, keepdim=True)
            features_norm = features / features.norm(dim=1, keepdim=True)
            
            # Calculate similarity matrix
            similarity = tta_emb_norm @ features_norm.T
            
            print("\nSimilarity Statistics:")
            print(f"Mean similarity: {similarity.mean().item():.4f}")
            print(f"Std similarity: {similarity.std().item():.4f}")
            print(f"Max similarity: {similarity.max().item():.4f}")
            print(f"Min similarity: {similarity.min().item():.4f}")
        
        # Make sure everything is on CPU before visualization
        combined_features = combined_features.detach().cpu()
        
        # Visualize embeddings using t-SNE
        self.visualize_embeddings(combined_features, labels, 
                                title="t-SNE Visualization of Features")

    def forward(self, x, optimizer):
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
                image_features = image_features_pre @ self.model.visual.proj
                image_features = image_features.squeeze(0)
                
        indexes, _ = self.infer_pseudo_labels(image_features)
        class_label = self.inverted_cls[indexes.item()]

        segments_gt, unique_labels = self.get_segments_gt(video_name, fps)

        for _ in range(self.steps):
            if self.image_projection:
                image_features = (image_features_pre @ self.model.visual.proj).squeeze(
                    0
                )
                before_optimization_parameters_image_encoder = copy.deepcopy(
                    self.model.visual.state_dict()
                )
                before_optimization_image_projection = copy.deepcopy(
                    self.model.visual.proj
                )

            if self.text_projection:
                before_optimization_text_projection = copy.deepcopy(
                    self.model.text.text_projection
                )
                before_optimization_parameters_text_encoder = copy.deepcopy(
                    self.model.text.state_dict()
                )
            else:
                before_optimization_parameters_text_encoder = copy.deepcopy(
                    self.model.text.state_dict()
                )
            before_optimization_logit_scale = copy.deepcopy(self.model.logit_scale)

            tta_emb = self.compute_tta_embedding(class_label, image_features.device)
            
            features = image_features - self.background_embedding if self.remove_background else image_features

            similarity = tta_emb @ features.T

            
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
            similarity_p = (
                tta_emb @ image_features_p.T
            ).unsqueeze(0)
            similarity_n = (
                tta_emb @ image_features_n.T
            ).unsqueeze(0)
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
            
            tta_emb = tta_emb.unsqueeze(0)
            if self.ltype in ["BYOL", "BCE"]:
                tta_loss = self.tta_loss(similarity, gt)
            elif self.ltype == "BYOLfeat":
                tta_loss = self.tta_loss(similarity, gt) + self.tta_loss(
                    image_features_p,
                    tta_emb.repeat_interleave(image_features_p.shape[0], dim=0),
                )
            elif self.ltype == "TTA":
                tta_loss = self.tta_loss(image_features, tta_emb, positive_indices=pindices)
            else:
                raise ValueError(f"Not implemented loss type: {self.ltype}")
            
            print(f"Step {_} - TTA Loss: {tta_loss.item():.4f}")

            tta_loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

           
            
        if self.text_projection:
            assert not torch.equal(
                before_optimization_text_projection,
                copy.deepcopy(self.model.text.text_projection),
            ), f"Parameter text_projection has not been updated."

        if self.image_projection:
            assert not torch.equal(
                before_optimization_image_projection,
                copy.deepcopy(self.model.visual.proj),
            ), f"Parameter has not been updated."

        #print(f"tta emb: {tta_emb.shape}")
        #print(f"image features: {image_features.shape}")


        # load avg features
        #image_features_classes = self.get_support_image_features(self.model).to(self.device)       

        #print(class_label)
        #print(f"indexes: {indexes.item()}")
        # check if the model collapses
        #self.check_for_collapse(image_features_classes, image_features.mean(dim=0).unsqueeze(0), labels=["0_TTA", "1_TTA", "2_TTA", "3_TTA", "4_TTA", "5_TTA", "6_TTA", "7_TTA", "8_TTA", "9_TTA", indexes.item()])

        
        
        
        with torch.no_grad():
            tta_emb = self.compute_tta_embedding(class_label, image_features.device)
            
            if self.remove_background:
                image_features = image_features - self.background_embedding
            
            image_features_norm = image_features / image_features.norm(
                dim=-1, keepdim=True
            )
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
            
            

            if segments:
                image_features = [
                    torch.mean(image_features[seg[0] : seg[1]], dim=0)
                    for seg in segments
                ]
                image_features_classes = self.get_support_image_features(self.model).to(self.device)
                image_features = torch.stack(image_features)
                
                
                print(f"shape image_features: {image_features.shape}")
                print(f"shape image_features_classes: {image_features_classes.shape}")
                pred, scores = self.compute_score(
                    image_features,
                    image_features_classes.to(image_features.device),
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
                    for i in range((len(segments)))
                ]
            else:
                output = [
                    {
                        "label": -1,
                        "score": 0,
                        "segment": [],
                    }
                ]
        self.model.text.load_state_dict(before_optimization_parameters_text_encoder)
        self.model.logit_scale = before_optimization_logit_scale
        if self.image_projection:
            self.model.visual.load_state_dict(
                before_optimization_parameters_image_encoder
            )

        print(f"Output: {output}")
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