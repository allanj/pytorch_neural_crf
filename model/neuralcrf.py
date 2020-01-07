# 
# @author: Allan
#

import torch
import torch.nn as nn

from config import START, STOP, PAD, log_sum_exp_pytorch
from model.charbilstm import CharBiLSTM
from model.bilstm_encoder import BiLSTMEncoder
from model.linear_crf_inferencer import LinearCRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import ContextEmb
from typing import Tuple
from overrides import overrides


class NNCRF(nn.Module):

    def __init__(self, config, print_info: bool = True):
        super(NNCRF, self).__init__()
        self.device = config.device
        self.encoder = BiLSTMEncoder(config, print_info=print_info)
        self.inferencer = LinearCRF(config, print_info=print_info)

    @overrides
    def forward(self, words: torch.Tensor,
                    word_seq_lens: torch.Tensor,
                    batch_context_emb: torch.Tensor,
                    chars: torch.Tensor,
                    char_seq_lens: torch.Tensor,
                    tags: torch.Tensor) -> torch.Tensor:
        """
        Calculate the negative loglikelihood.
        :param words: (batch_size x max_seq_len)
        :param word_seq_lens: (batch_size)
        :param batch_context_emb: (batch_size x max_seq_len x context_emb_size)
        :param chars: (batch_size x max_seq_len x max_char_len)
        :param char_seq_lens: (batch_size x max_seq_len)
        :param tags: (batch_size x max_seq_len)
        :return: the total negative log-likelihood loss
        """
        lstm_scores = self.encoder(words, word_seq_lens, batch_context_emb, chars, char_seq_lens)
        batch_size = words.size(0)
        sent_len = words.size(1)
        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long).view(1, sent_len).expand(batch_size, sent_len).to(self.device)
        mask = torch.le(maskTemp, word_seq_lens.view(batch_size, 1).expand(batch_size, sent_len)).to(self.device)
        unlabed_score, labeled_score =  self.inferencer(lstm_scores, word_seq_lens, tags, mask)
        return unlabed_score - labeled_score

    def decode(self, batchInput: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        :param batchInput:
        :return:
        """
        wordSeqTensor, wordSeqLengths, batch_context_emb, charSeqTensor, charSeqLengths, tagSeqTensor = batchInput
        features = self.encoder(wordSeqTensor, wordSeqLengths, batch_context_emb,charSeqTensor,charSeqLengths)
        bestScores, decodeIdx = self.inferencer.decode(features, wordSeqLengths)
        return bestScores, decodeIdx
