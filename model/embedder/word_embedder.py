import torch
import torch.nn as nn
from model.module.charbilstm import CharBiLSTM
from config import ContextEmb

class WordEmbedder(nn.Module):


    def __init__(self, config, print_info=False):
        """
        This word embedder allows to static contextualized representation.
        :param config:
        :param print_info:
        """
        super(WordEmbedder, self).__init__()
        self.static_context_emb = config.static_context_emb
        self.use_char = config.use_char_rnn
        self.device = config.device
        if self.use_char:
            self.char_feature = CharBiLSTM(config, print_info=print_info)

        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(config.word_embedding), freeze=False).to(self.device)
        self.word_drop = nn.Dropout(config.dropout).to(self.device)

        self.output_size = config.embedding_dim
        if self.static_context_emb != ContextEmb.none:
            self.output_size += config.context_emb_size
        if self.use_char:
            self.output_size += config.charlstm_hidden_dim

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
            word_emb = torch.cat([word_emb, context_emb.to(self.device)], 2)
        if self.use_char:
            char_features = self.char_feature(chars, char_seq_lens)
            word_emb = torch.cat([word_emb, char_features], 2)

        word_rep = self.word_drop(word_emb)
        return word_rep
