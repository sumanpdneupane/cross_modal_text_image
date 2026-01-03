import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import DistilBertTokenizer
from transformers import DistilBertModel
from src.utils.utils import chunk_text

class TextEncoder(nn.Module):
    def __init__(self, embed_dim=1024, dropout=0.1, device="cpu", max_chunk_length=128):
        super().__init__()
        self.device = device
        self.max_chunk_length = max_chunk_length

        # DistilBERT
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        hidden_dim = self.bert.config.hidden_size  # 768

        # Projection head (MATCHES ImageEncoder)
        self.projection = nn.Linear(hidden_dim, embed_dim)
        self.activation = nn.GELU()
        self.linear = nn.Linear(embed_dim, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.to(device)

    def forward(self, texts):
        batch_embeddings = []

        # ---- Chunk texts ----
        all_chunks = []
        chunk_map = []
        for idx, text in enumerate(texts):
            chunks = chunk_text(text, self.max_chunk_length)
            all_chunks.extend(chunks)
            chunk_map.extend([idx] * len(chunks))

        # ---- Tokenize ----
        encoded = self.tokenizer(
            all_chunks,
            padding=True,
            truncation=True,
            max_length=self.max_chunk_length,
            return_tensors="pt"
        ).to(self.device)

        # ---- BERT forward ----
        outputs = self.bert(**encoded)
        chunk_embeddings = outputs.last_hidden_state.mean(dim=1)

        # ---- Pool chunks per text ----
        chunk_map = torch.tensor(chunk_map, device=self.device)
        for i in range(len(texts)):
            pooled = chunk_embeddings[chunk_map == i].mean(dim=0)
            batch_embeddings.append(pooled)

        x = torch.stack(batch_embeddings)

        # ---- Projection head ----
        x = self.projection(x)
        x = self.activation(x)
        x = self.linear(x)
        x = self.norm(x)
        x = self.dropout(x)

        return F.normalize(x, dim=1)

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=1024, dropout=0.1, device="cpu", pretrained=True):
        super().__init__()
        self.device = device

        # ResNet-50 backbone
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 # DEFAULT == V2
            if pretrained else None
        )

        # Remove classifier
        self.backbone.fc = nn.Identity()
        backbone_dim = 2048

        # Projection head (MATCHES TextEncoder)
        self.projection = nn.Linear(backbone_dim, embed_dim)
        self.activation = nn.GELU()
        self.linear = nn.Linear(embed_dim, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.to(device)

    def forward(self, images):
        x = self.backbone(images)

        x = self.projection(x)
        x = self.activation(x)
        x = self.linear(x)
        x = self.norm(x)
        x = self.dropout(x)

        return F.normalize(x, dim=1)


class FusionModel(nn.Module):
    def __init__(self, text_embed_dim=1024, long_embed_dim=1024, image_embed_dim=1024, device="cpu"):
        super().__init__()
        self.device = device
        self.text_encoder = TextEncoder(embed_dim=text_embed_dim, device=device)
        self.long_encoder = TextEncoder(embed_dim=long_embed_dim, device=device)
        self.image_encoder = ImageEncoder(embed_dim=image_embed_dim, device=device)

    def forward_text(self, texts):
        return self.text_encoder(texts)

    def forward_long_text(self, texts):
        return self.long_encoder(texts)

    def forward_image(self, images):
        return self.image_encoder(images)


# Loss Model
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, emb_q, emb_k):
        # L2 normalize
        emb_q = F.normalize(emb_q, dim=1)
        emb_k = F.normalize(emb_k, dim=1)

        # Similarity matrix [B, B]
        sim_matrix = torch.matmul(emb_q, emb_k.T) / self.temperature

        # Ground-truth labels: diagonal
        labels = torch.arange(emb_q.size(0), device=emb_q.device)

        # Cross-entropy loss in both directions (symmetric)
        loss_qk = F.cross_entropy(sim_matrix, labels)       # query -> key
        loss_kq = F.cross_entropy(sim_matrix.T, labels)     # key -> query

        return (loss_qk + loss_kq) / 2