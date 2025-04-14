import torch
import torch.nn as nn

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings):
        batch_size, n_pairs, two, embedding_dim = embeddings.shape
        if two != 2:
            raise ValueError(f"Expected 2 views per positive pair, got {two}.")
        
        # Normalize embeddings
        all_embeddings = embeddings.view(-1, embedding_dim)
        all_embeddings = nn.functional.normalize(all_embeddings, dim=1)

        # Compute similarity matrix and mask out self-similarity
        total_patches = batch_size * n_pairs * two
        similarity_matrix = torch.matmul(all_embeddings, all_embeddings.t()) / self.temperature
        mask_self = torch.eye(total_patches, dtype=torch.bool, device=embeddings.device)
        similarity_matrix = similarity_matrix.masked_fill(mask_self, float('-inf'))

        # Get target labels
        labels = torch.arange(total_patches, device=embeddings.device)
        labels[0::2] += 1
        labels[1::2] -= 1

        loss = nn.functional.cross_entropy(
            similarity_matrix, labels, reduction="mean"
        )

        return loss