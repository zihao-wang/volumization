# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class LSTMClassifier(nn.Module):
    def __init__(self,
                 output_size=2,
                 hidden_size=256,
                 weights=None):
        """
        output_size : num of class
        hidden_size : hyper-params of LSTM
        weights : pretrained word embedding
        """
        super(LSTMClassifier, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        vocab_size, embedding_len = weights.shape
        self.word_embeddings = nn.Embedding(vocab_size, embedding_len)
        if weights: self.word_embeddings.from_pretrained(weights, freeze=True)
        self.lstm = nn.LSTM(embedding_len, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    def forward(self, input_sentence):
        batch_size, num_tokens = input_sentence.shape
        device = input_sentence.device
        input = self.word_embeddings(input_sentence)  # (batch_size, num_tokens,  embedding_length)
        input = input.permute(1, 0, 2)  # (num_sequences, batch_size, embedding_length)
        h_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        final_output = self.label(final_hidden_state[-1])
        return final_output
