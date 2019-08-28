# 
# @author: Allan
#

import torch
import torch.nn as nn

from config import START, STOP, PAD, log_sum_exp_pytorch, Config
from model.charbilstm import CharBiLSTM
from model.bilstm_encoder import BiLSTMEncoder
from model.linear_crf_inferencer import LinearCRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import ContextEmb
from typing import Tuple
from overrides import overrides


class NNCRF(nn.Module):

    def __init__(self, config: Config,
                 print_info: bool = True):
        super(NNCRF, self).__init__()
        self.device = config.device
        self.encoder = BiLSTMEncoder(config, print_info=print_info)
        self.inferencer = None
        if config.use_crf_layer:
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
        :return: the loss with shape (batch_size)
        """
        batch_size = words.size(0)
        max_sent_len = words.size(1)
        #Shape: (batch_size, max_seq_len, num_labels)
        lstm_scores = self.encoder(words, word_seq_lens, batch_context_emb, chars, char_seq_lens)
        maskTemp = torch.arange(1, max_sent_len + 1, dtype=torch.long).view(1, max_sent_len).expand(batch_size, max_sent_len).to(self.device)
        mask = torch.le(maskTemp, word_seq_lens.view(batch_size, 1).expand(batch_size, max_sent_len)).to(self.device)
        if self.inferencer is not None:
            unlabed_score, labeled_score =  self.inferencer(lstm_scores, word_seq_lens, tags, mask)
            loss = unlabed_score - labeled_score
        else:
            loss = self.compute_nll_loss(lstm_scores, tags, mask, word_seq_lens)
        return loss

    def decode(self, batchInput: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        :param batchInput:
        :return:
        """
        wordSeqTensor, wordSeqLengths, batch_context_emb, charSeqTensor, charSeqLengths, tagSeqTensor = batchInput
        lstm_scores = self.encoder(wordSeqTensor, wordSeqLengths, batch_context_emb,charSeqTensor,charSeqLengths)
        if self.inferencer is not None:
            bestScores, decodeIdx = self.inferencer.decode(lstm_scores, wordSeqLengths)
        else:
            bestScores, decodeIdx = torch.max(lstm_scores, dim=2)
        return bestScores, decodeIdx

    def compute_nll_loss(self, candidate_scores, target, mask, word_seq_lens):
        """
        Directly compute the loss right after the linear layer instead of CRF layer.
        Partially taken from `masked_cross_entropy.py` (https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1)
        :param candidate_scores:
        :param target:
        :param mask:
        :param word_seq_lens:
        :return:
        """
        # logits_flat: (batch * max_len, num_classes)
        logits_flat = candidate_scores.view(-1, candidate_scores.size(-1))
        # log_probs_flat: (batch * max_len, num_classes)
        log_probs_flat = torch.log_softmax(logits_flat, dim=1)
        # target_flat: (batch * max_len, 1)
        target_flat = target.view(-1, 1)
        # losses_flat: (batch * max_len, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        # losses: (batch, max_len)
        losses = losses_flat.view(*target.size())
        # # mask: (batch, max_len)
        # mask = _sequence_mask(sequence_length=length, max_len=target.size(1))
        losses = losses * mask.float()
        # loss = losses.sum() / word_seq_lens.float().sum()
        loss = losses.sum()
        return loss