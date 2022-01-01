
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTMEncoder(nn.Module):
    """
    BILSTM encoder.
    output the score of all labels.
    """

    def __init__(self, label_size: int, input_dim:int,
                 hidden_dim: int,
                 drop_lstm:float=0.5,
                 num_lstm_layers: int =1):
        super(BiLSTMEncoder, self).__init__()

        self.label_size = label_size
        print("[Model Info] Input size to LSTM: {}".format(input_dim))
        print("[Model Info] LSTM Hidden Size: {}".format(hidden_dim))
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, num_layers=num_lstm_layers, batch_first=True, bidirectional=True)
        self.drop_lstm = nn.Dropout(drop_lstm)
        self.hidden2tag = nn.Linear(hidden_dim, self.label_size)

    def forward(self, word_rep: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Encoding the input with BiLSTM
        :param word_rep: (batch_size, sent_len, input rep size)
        :param word_seq_lens: (batch_size, 1)
        :return: emission scores (batch_size, sent_len, hidden_dim)
        """
        sorted_seq_len, permIdx = word_seq_lens.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[permIdx]

        packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len.cpu(), True)
        lstm_out, _ = self.lstm(packed_words, None)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  ## CARE: make sure here is batch_first, otherwise need to transpose.
        feature_out = self.drop_lstm(lstm_out)

        outputs = self.hidden2tag(feature_out)
        return outputs[recover_idx]


