import torch
import torch.nn.functional as F

def info_nce_logits(
    features: torch.Tensor,
    n_views: int = 2,
    temperature: float = 1.0,
):
    """ Prepare logits and labels for InfoNCE loss. """
    
    # features: (n_views * B, hidden_dim)
    num_samples, B = features.shape[0] // n_views, features.shape[0]
    
    labels = torch.cat(
        [torch.arange(num_samples) for _ in range(n_views)], dim=0
    )
    labels = (labels[:, None] == labels[None, :]).float().to(features.device)
    
    features = F.normalize(features, dim=-1)
    s_matrix = torch.matmul(features, features.T)
    
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(
        labels.shape[0], dtype=torch.bool, device=labels.device
    )
    labels = labels[~mask].view(B, -1)
    s_matrix = s_matrix[~mask].view(B, -1)
    
    pos = s_matrix[labels.bool()].view(B, -1)
    neg = s_matrix[~labels.bool()].view(B, -1)
    
    logits = torch.cat([pos, neg], dim=1) / temperature
    labels = torch.zeros(B, dtype=torch.long).to(features.device)
    
    return logits, labels
    
    