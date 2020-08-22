
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from overrides import overrides

class LinearEncoder(nn.Module):

    def __init__(self, config, input_dim, print_info: bool = True):
        super(LinearEncoder, self).__init__()

        final_hidden_dim = input_dim
        self.label_size = config.label_size
        self.hidden2tag = nn.Linear(final_hidden_dim, self.label_size)

    @overrides
    def forward(self, word_rep: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Encoding the input with BiLSTM
        :param word_rep: (batch_size, sent_len, input rep size)
        :param word_seq_lens: (batch_size, 1)
        :return: emission scores (batch_size, sent_len, hidden_dim)
        """
        outputs = self.hidden2tag(word_rep)
        return outputs


