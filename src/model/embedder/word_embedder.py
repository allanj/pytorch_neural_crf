import torch
import torch.nn as nn
from src.model.module.charbilstm import CharBiLSTM
from src.config import ContextEmb
import numpy as np

class WordEmbedder(nn.Module):


    def __init__(self, word_embedding: np.ndarray,
                 embedding_dim: int,
                 static_context_emb: ContextEmb,
                 context_emb_size: int,
                 use_char_rnn: bool,
                 char_emb_size: int,
                 char_size:int,
                 char_hidden_size: int,
                 dropout:float = 0.5):
        """
        This word embedder allows to static contextualized representation.
        :param config:
        :param print_info:
        """
        super(WordEmbedder, self).__init__()
        self.static_context_emb = static_context_emb
        self.use_char = use_char_rnn
        if self.use_char:
            self.char_feature = CharBiLSTM(char_emb_size=char_emb_size, char_size=char_size, char_hidden_size=char_hidden_size,
                                           drop_char=dropout)

        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word_embedding), freeze=False)
        self.word_drop = nn.Dropout(dropout)

        self.output_size = embedding_dim
        if self.static_context_emb != ContextEmb.none:
            self.output_size += context_emb_size
        if self.use_char:
            self.output_size += char_hidden_size

    def get_output_dim(self):
        return self.output_size

    def forward(self, words: torch.Tensor,
                       word_seq_lens: torch.Tensor,
                       context_emb: torch.Tensor,
                       chars: torch.Tensor,
                       char_seq_lens: torch.Tensor) -> torch.Tensor:
        """
            Encoding the input with embedding
            :param word_seq_tensor: (batch_size, sent_len)   NOTE: The word seq actually is already ordered before come here.
            :param word_seq_lens: (batch_size, 1)
            :param batch_context_emb: (batch_size, sent_len, context embedding) ELMo embedings
            :param char_inputs: (batch_size * sent_len * word_length)
            :param char_seq_lens: numpy (batch_size * sent_len , 1)
            :return: word representation (batch_size, sent_len, hidden_dim)
        """
        word_emb = self.word_embedding(words)
        if self.static_context_emb != ContextEmb.none:
            dev_num = word_emb.get_device()
            curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
            word_emb = torch.cat([word_emb, context_emb.to(curr_dev)], 2)
        if self.use_char:
            char_features = self.char_feature(chars, char_seq_lens.cpu())
            word_emb = torch.cat([word_emb, char_features], 2)

        word_rep = self.word_drop(word_emb)
        return word_rep
