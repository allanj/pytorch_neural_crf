# 
# @author: Allan
#

import torch
import torch.nn as nn

from src.model.module.bilstm_encoder import BiLSTMEncoder
from src.model.module.linear_crf_inferencer import LinearCRF
from src.model.embedder import WordEmbedder
from typing import Tuple
from overrides import overrides


class NNCRF(nn.Module):

    def __init__(self, config):
        super(NNCRF, self).__init__()
        self.embedder = WordEmbedder(word_embedding=config.word_embedding,
                                     embedding_dim=config.embedding_dim,
                                     static_context_emb=config.static_context_emb,
                                     context_emb_size=config.context_emb_size,
                                     use_char_rnn=config.use_char_rnn,
                                     char_emb_size=config.char_emb_size,
                                     char_size=len(config.char2idx),
                                     char_hidden_size=config.charlstm_hidden_dim,
                                     dropout=config.dropout)
        self.encoder = BiLSTMEncoder(label_size=config.label_size,
                                     input_dim=self.embedder.get_output_dim(),
                                     hidden_dim=config.hidden_dim,
                                     drop_lstm=config.dropout)
        self.inferencer = LinearCRF(label_size=config.label_size, label2idx=config.label2idx, add_iobes_constraint=config.add_iobes_constraint,
                                    idx2labels=config.idx2labels)

    @overrides
    def forward(self, words: torch.Tensor,
                    word_seq_lens: torch.Tensor,
                    context_emb: torch.Tensor,
                    chars: torch.Tensor,
                    char_seq_lens: torch.Tensor,
                    labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate the negative loglikelihood.
        :param words: (batch_size x max_seq_len)
        :param word_seq_lens: (batch_size)
        :param context_emb: (batch_size x max_seq_len x context_emb_size)
        :param chars: (batch_size x max_seq_len x max_char_len)
        :param char_seq_lens: (batch_size x max_seq_len)
        :param labels: (batch_size x max_seq_len)
        :return: the total negative log-likelihood loss
        """
        word_rep = self.embedder(words, word_seq_lens, context_emb, chars, char_seq_lens)
        lstm_scores = self.encoder(word_rep, word_seq_lens.cpu())
        batch_size = words.size(0)
        sent_len = words.size(1)

        dev_num = word_rep.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long, device=curr_dev).view(1, sent_len).expand(batch_size, sent_len)
        mask = torch.le(maskTemp, word_seq_lens.view(batch_size, 1).expand(batch_size, sent_len))
        unlabed_score, labeled_score = self.inferencer(lstm_scores, word_seq_lens, labels, mask)
        return unlabed_score - labeled_score

    def decode(self, words: torch.Tensor,
                    word_seq_lens: torch.Tensor,
                    context_emb: torch.Tensor,
                    chars: torch.Tensor,
                    char_seq_lens: torch.Tensor,
                    **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        :param batchInput:
        :return:
        """
        word_rep = self.embedder(words, word_seq_lens, context_emb, chars, char_seq_lens)
        features = self.encoder(word_rep, word_seq_lens.cpu())
        bestScores, decodeIdx = self.inferencer.decode(features, word_seq_lens)
        return bestScores, decodeIdx
