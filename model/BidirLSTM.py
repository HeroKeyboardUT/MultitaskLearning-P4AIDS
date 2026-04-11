import torch
import torch.nn as nn


class BidirLSTM(nn.Module):
	"""
	Bidirectional LSTM for multi-task classification.

	This model predicts both sentiment and topic labels from the same encoded text.
	All output dimensions and hyperparameters are configurable via constructor args.
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
	) -> None:
		super().__init__()

		self.embedding = nn.Embedding(
			num_embeddings=vocab_size,
			embedding_dim=embedding_dim,
			padding_idx=padding_idx,
		)

		# PyTorch LSTM only applies internal dropout when n_layers > 1.
		lstm_dropout = dropout if n_layers > 1 else 0.0
		self.lstm = nn.LSTM(
			input_size=embedding_dim,
			hidden_size=hidden_dim,
			num_layers=n_layers,
			bidirectional=bidirectional,
			batch_first=True,
			dropout=lstm_dropout,
		)

		direction_factor = 2 if bidirectional else 1
		head_input_dim = hidden_dim * direction_factor

		self.fc_sentiment = nn.Linear(head_input_dim, num_sentiment_labels+1)
		self.fc_topic = nn.Linear(head_input_dim, num_topic_labels+1)
		self.dropout = nn.Dropout(dropout)
		self.bidirectional = bidirectional

	def forward(
		self,
		text: torch.Tensor,
		text_lengths: torch.Tensor,
		return_hidden: bool = False,
	):
		"""
		Forward pass for multi-task prediction.

		Args:
			text: Token ids with shape [batch_size, seq_len].
			text_lengths: Sequence lengths with shape [batch_size].
			return_hidden: If True, return shared hidden representation.

		Returns:
			Tuple of (sentiment_logits, topic_logits) or
			(sentiment_logits, topic_logits, hidden_repr) when return_hidden=True.
		"""
		embedded = self.dropout(self.embedding(text))

		packed_embedded = nn.utils.rnn.pack_padded_sequence(
			embedded,
			text_lengths.to("cpu"),
			batch_first=True,
			enforce_sorted=False,
		)

		_, (hidden, _) = self.lstm(packed_embedded)

		if self.bidirectional:
			# hidden shape: [num_layers * 2, batch_size, hidden_dim]
			shared_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
		else:
			# hidden shape: [num_layers, batch_size, hidden_dim]
			shared_hidden = hidden[-1, :, :]

		shared_hidden = self.dropout(shared_hidden)

		sent_logits = self.fc_sentiment(shared_hidden)
		topic_logits = self.fc_topic(shared_hidden)

		if return_hidden:
			return sent_logits, topic_logits, shared_hidden

		return sent_logits, topic_logits
