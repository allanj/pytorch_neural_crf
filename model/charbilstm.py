# 
# @author: Allan
#
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CharBiLSTM(nn.Module):

    def __init__(self, config):
        super(CharBiLSTM, self).__init__()
        print("[Info] Building character-level LSTM")
        self.char_emb_size = config.char_emb_size
        self.char2idx = config.char2idx
        self.chars = config.idx2char
        self.char_size = len(self.chars)
        self.device = config.device
        self.hidden = config.charlstm_hidden_dim
        self.dropout = nn.Dropout(config.dropout).to(self.device)
        self.char_embeddings = nn.Embedding(self.char_size, self.char_emb_size)
        # self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(self.char_size, self.char_emb_size)))
        self.char_embeddings = self.char_embeddings.to(self.device)

        self.char_lstm = nn.LSTM(self.char_emb_size, self.hidden // 2 ,num_layers=1, batch_first=True, bidirectional=True).to(self.device)


    def get_last_hiddens(self, char_seq_tensor, char_seq_len):
        """
            input:
                char_seq_tensor: (batch_size, sent_len, word_length)
                char_seq_len: (batch_size, sent_len)
            output:
                Variable(batch_size, sent_len, char_hidden_dim )
        """
        batch_size = char_seq_tensor.size(0)
        sent_len = char_seq_tensor.size(1)
        char_seq_tensor = char_seq_tensor.view(batch_size * sent_len, -1)
        char_seq_len = char_seq_len.view(batch_size * sent_len)
        sorted_seq_len, permIdx = char_seq_len.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = char_seq_tensor[permIdx]

        char_embeds = self.dropout(self.char_embeddings(sorted_seq_tensor))
        pack_input = pack_padded_sequence(char_embeds, sorted_seq_len, batch_first=True)

        _, char_hidden = self.char_lstm(pack_input, None)  ###
        ## char_hidden = (h_t, c_t)
        #  char_hidden[0] = h_t = (2, batch_size, lstm_dimension)
        # char_rnn_out, _ = pad_packed_sequence(char_rnn_out)
        ## transpose because the first dimension is num_direction x num-layer
        hidden = char_hidden[0].transpose(1,0).contiguous().view(batch_size * sent_len, 1, -1)   ### before view, the size is ( batch_size * sent_len, 2, lstm_dimension) 2 means 2 direciton..
        return hidden[recover_idx].view(batch_size, sent_len, -1)



    def forward(self, char_input, seq_lengths):
        return self.get_last_hiddens(char_input, seq_lengths)



