from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

ce = nn.CrossEntropyLoss()
bce_cls = nn.BCEWithLogitsLoss()


class ByolLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, online_proj, target_proj):
        """
        Args:
            online_proj: Projections from online network [B, D]
            target_proj: Projections from target network [B, D]
        Returns:
            loss: BYOL's MSE-style loss
        """


        # Normalize projections
        online_proj = F.normalize(online_proj, dim=-1)
        target_proj = F.normalize(target_proj, dim=-1).detach()  # Stop gradient

        # Original BYOL loss: 2 - 2 * cosine similarity (equivalent to MSE)
        loss = 2 - 2 * (online_proj * target_proj).sum(dim=-1).mean()
        
        return loss

def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=0)
    return entropy

class AdaptiveMarginLoss(nn.Module):
    def __init__(self, margin=0.5, gamma=2):
        super().__init__()
        self.margin = margin
        self.gamma = gamma  # Imbalance focus parameter

    def forward(self, pos_similarities, neg_similarities):
        """pos_similarities: [N_pos]
           neg_similarities: [N_neg]"""
        pos_loss = torch.relu(1 - pos_similarities).pow(self.gamma)
        neg_loss = torch.relu(neg_similarities - self.margin).pow(self.gamma)
        return (pos_loss.mean() + neg_loss.mean()) / 2

class ProtoNCELoss(nn.Module):
    def __init__(self, temp=0.1):
        super().__init__()
        self.temp = temp

    def forward(self, query_features, prototypes):
        """query_features: [N_segs, D]
           prototypes: [N_classes, D]"""
        sim = query_features @ prototypes.T  # [N_segs, N_classes]
        sim /= self.temp
        targets = torch.zeros(sim.shape[0]).long().to(sim.device)  # All pseudo-class
        
        # For multi-class use: targets = pseudo_labels 
        return F.cross_entropy(sim, targets)
    

class HybridLoss(nn.Module):
    def __init__(self, margin=0.3, temp=0.2, alpha=0.7):
        super().__init__()
        self.margin_loss = AdaptiveMarginLoss(margin)
        self.proto_loss = ProtoNCELoss(temp)
        self.alpha = alpha  # Weighting factor

    def forward(self, pos_sim, neg_sim, features, prototypes):
        return (self.alpha * self.margin_loss(pos_sim, neg_sim) + 
                (1-self.alpha) * self.proto_loss(features, prototypes))
    
class TTALoss(nn.Module):
    def __init__(self, confidence_threshold=0.9, entropy_weight=1.0, batch_size=128):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.entropy_weight = entropy_weight
        self.batch_size = batch_size  # For processing feature diversity in chunks
    
    def forward(self, logits, features=None):
        """
        Memory-efficient implementation of test-time adaptation objectives
        
        Args:
            logits: Model predictions [B, num_classes]
            features: Optional feature vectors [B, D]
        """
        # Get softmax probabilities with memory-efficient computation
        probs = F.softmax(logits, dim=1)
        
        # 1. Entropy minimization with confidence thresholding
        entropy = -(probs * torch.log(probs + 1e-6)).sum(dim=1)
        confidence, predictions = probs.max(dim=1)
        mask = confidence > self.confidence_threshold
        
        # Compute entropy loss efficiently
        entropy_loss = entropy[mask].mean() if mask.sum() > 0 else entropy.mean()
            
        # 2. Feature diversity with batch processing
        diversity_loss = 0
        if features is not None:
            features = F.normalize(features.detach(), dim=1)  # Detach to save memory
            B = features.shape[0]
            
            # Process feature diversity in chunks to save memory
            chunk_diversity = []
            for i in range(0, B, self.batch_size):
                chunk = features[i:min(i + self.batch_size, B)]
                # Compute similarity only for the chunk
                chunk_sim = torch.mm(chunk, chunk.t())
                chunk_div = (chunk_sim - torch.eye(chunk_sim.shape[0], 
                                                 device=chunk_sim.device)).pow(2).mean()
                chunk_diversity.append(chunk_div)
            
            diversity_loss = torch.stack(chunk_diversity).mean()
            
        # 3. Pseudo-label consistency
        pseudo_labels = predictions  # No need for clone() since we're not modifying
        pseudo_loss = F.cross_entropy(logits, pseudo_labels, reduction='none')
        pseudo_loss = (pseudo_loss * confidence).mean()
        
        # Free unnecessary tensors explicitly
        del confidence, predictions, entropy
        
        # Combine losses with reduced weight on diversity
        total_loss = pseudo_loss + self.entropy_weight * entropy_loss + 0.1 * diversity_loss
        
        metrics = {
            'entropy_loss': entropy_loss.item(),
            'pseudo_loss': pseudo_loss.item(),
            'diversity_loss': diversity_loss.item() if features is not None else 0,
            'confident_samples': mask.sum().item()
        }
        
        return total_loss, metrics

# Optional: Memory-efficient consistency regularization
class ConsistencyRegularization(nn.Module):
    def __init__(self, consistency_weight=1.0):
        super().__init__()
        self.consistency_weight = consistency_weight
    
    def forward(self, orig_logits, aug_logits):
        """Memory-efficient consistency regularization"""
        # Compute KL divergence in a memory-efficient way
        log_probs = F.log_softmax(aug_logits, dim=1)
        with torch.no_grad():  # Don't store intermediate tensors for original probs
            orig_probs = F.softmax(orig_logits, dim=1)
        
        consistency_loss = -(orig_probs * log_probs).sum(dim=1).mean()
        return self.consistency_weight * consistency_loss
    

class ImprovedTTALoss(nn.Module):
    def __init__(self, margin=0.5, temp=0.1, consistency_weight=0.5):
        super().__init__()
        self.margin = margin
        self.temp = temp
        self.consistency_weight = consistency_weight
    
    def forward(self, tta_emb, image_features, background_emb=None):
        """
        Improved TTA loss combining contrastive learning with consistency regularization
        
        Args:
            tta_emb: Target embedding [1, D]
            image_features: Image features [N, D]
            background_emb: Optional background embedding to subtract [1, D]
        """
        # Normalize embeddings
        tta_emb = F.normalize(tta_emb, dim=-1)
        image_features = F.normalize(image_features, dim=-1)
        
        if background_emb is not None:
            background_emb = F.normalize(background_emb, dim=-1)
            image_features = image_features - background_emb
            image_features = F.normalize(image_features, dim=-1)
        
        # Compute similarities
        similarities = (tta_emb @ image_features.T).squeeze()  # [N]
        
        # Get positive and negative masks based on similarity scores
        pos_mask = similarities > similarities.mean() + self.margin
        neg_mask = similarities < similarities.mean() - self.margin
        
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0).to(similarities.device)
        
        # Contrastive loss
        pos_sims = similarities[pos_mask]
        neg_sims = similarities[neg_mask]
        
        pos_term = -torch.log(torch.sigmoid(pos_sims / self.temp)).mean()
        neg_term = -torch.log(1 - torch.sigmoid(neg_sims / self.temp)).mean()
        contrastive_loss = pos_term + neg_term
        
        # Consistency regularization
        pos_features = image_features[pos_mask]
        pos_centroid = pos_features.mean(dim=0, keepdim=True)
        consistency_loss = (1 - F.cosine_similarity(pos_features, pos_centroid)).mean()
        
        # Combined loss
        total_loss = contrastive_loss + self.consistency_weight * consistency_loss
        
        return total_loss