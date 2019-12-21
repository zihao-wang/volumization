# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


class AttentionModel(torch.nn.Module):
    def __init__(self,
                 output_size=2,
                 hidden_size=256,
                 weights=None):
        """
        output_size : num of class
        hidden_size : hyper-params of LSTM
        weights : pretrained word embedding
        """
        super(AttentionModel, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        vocab_size, embedding_len = weights.shape
        self.word_embeddings = nn.Embedding(vocab_size, embedding_len)
        if weights is not None: self.word_embeddings.from_pretrained(weights, freeze=True)
        self.lstm = nn.LSTM(embedding_len, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    def attention_net(self, lstm_output, final_state):

        """
        Now we will incorporate Attention mechanism in our LSTM model.
        In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM.
        We will be using torch.bmm for the batch matrix multiplication.

        Arguments
        ---------

        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM

        ---------

        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.

        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)

        """

        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, input_sentences, batch_size=None):

        """
        Parameters
        ----------
        input_sentences: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class which receives its input as the new_hidden_state which is basically the output of the Attention network.
        final_output.shape = (batch_size, output_size)

        """

        batch_size, num_tokens = input_sentences.shape
        device = input_sentences.device
        input = self.word_embeddings(input_sentences)  # (batch_size, num_tokens,  embedding_length)
        input = input.permute(1, 0, 2)  # (num_sequences, batch_size, embedding_length)
        h_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0)) # final_hidden_state.size() = (1, batch_size, hidden_size)
        output = output.permute(1, 0, 2) # output.size() = (batch_size, num_seq, hidden_size)

        attn_output = self.attention_net(output, final_hidden_state)
        logits = self.label(attn_output)

        return logits
