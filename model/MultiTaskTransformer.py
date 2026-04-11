import torch
import torch.nn as nn


class MultiTaskTransformer(nn.Module):
    """
    Multi-task Transformer encoder for sentiment and topic classification.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        num_sentiment_labels: int,
        num_topic_labels: int,
        padding_idx: int = 0,
        dropout: float = 0.3,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        self.fc_sentiment = nn.Linear(embedding_dim, num_sentiment_labels)
        self.fc_topic = nn.Linear(embedding_dim, num_topic_labels+1)

    def forward(self, text: torch.Tensor, attention_mask: torch.Tensor):
        # text shape: [batch_size, seq_len]
        batch_size, seq_len = text.shape

        positions = torch.arange(seq_len, device=text.device).unsqueeze(0).expand(batch_size, -1)

        x = self.dropout(self.embedding(text) + self.pos_embedding(positions))
        padding_mask = attention_mask == 0

        encoded = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = self.dropout(pooled)

        sent_logits = self.fc_sentiment(pooled)
        topic_logits = self.fc_topic(pooled)

        return sent_logits, topic_logits
