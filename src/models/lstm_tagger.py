""" encoder: lstm
    decoder: crf
"""
from src.models.base_model import SequenceLabelingModel
import torch.nn as nn


class BiLSTMSequentialLabelingModel(SequenceLabelingModel):
    def __init__(self, config) -> None:
        super(BiLSTMSequentialLabelingModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                embedding_dim=config.embedding_dim)
        self.lstm = nn.LSTM(config.embedding_dim,
                            config.hidden_size,
                            config.num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=config.dropout)
        self.classifier = nn.Linear(config.hidden_size * 2,
                            config.num_labels)

    def _init_weight(self) -> None:
        for name, w in self.named_parameters():
            if 'embedding' not in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(w)
                elif 'bias' in name:
                    nn.init.constant_(w, 0)
                else:
                    pass
    
    def forward(self, input):
        x = self.embedding(input)  # [batch_size, seq_len, embedding_dim]

        x, t = self.lstm(x)  # [batch_size, seq_len, hidden_size * 2]
        output = self.classifier(x)  # [batch_size, seq_len, num_classes]
        return output