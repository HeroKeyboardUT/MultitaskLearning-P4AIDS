import torch
import torch.nn as nn


class BidirRNN(nn.Module):
    """
    Bidirectional RNN-family encoder (RNN/GRU) for multi-task classification.

    This model predicts both sentiment and topic labels from the same encoded text.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        n_layers: int,
        num_sentiment_labels: int,
        num_topic_labels: int,
        padding_idx: int = 0,
        dropout: float = 0.5,
        bidirectional: bool = True,
        rnn_type: str = "gru",
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        rnn_dropout = dropout if n_layers > 1 else 0.0
        rnn_type = rnn_type.lower()

        if rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                batch_first=True,
                dropout=rnn_dropout,
            )
        elif rnn_type == "rnn":
            self.rnn = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                nonlinearity="tanh",
                bidirectional=bidirectional,
                batch_first=True,
                dropout=rnn_dropout,
            )
        else:
            raise ValueError("rnn_type must be either 'gru' or 'rnn'.")

        direction_factor = 2 if bidirectional else 1
        head_input_dim = hidden_dim * direction_factor

        self.fc_sentiment = nn.Linear(head_input_dim, num_sentiment_labels)
        self.fc_topic = nn.Linear(head_input_dim, num_topic_labels)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        return_hidden: bool = False,
    ):
        embedded = self.dropout(self.embedding(text))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            text_lengths.to("cpu"),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.rnn(packed_embedded)

        if isinstance(hidden, tuple):
            hidden = hidden[0]

        if self.bidirectional:
            shared_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            shared_hidden = hidden[-1, :, :]

        shared_hidden = self.dropout(shared_hidden)

        sent_logits = self.fc_sentiment(shared_hidden)
        topic_logits = self.fc_topic(shared_hidden)

        if return_hidden:
            return sent_logits, topic_logits, shared_hidden

        return sent_logits, topic_logits
