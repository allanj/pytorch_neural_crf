#
# @author: Allan
#

import torch
import torch.nn as nn

from src.model.module.bilstm_encoder import BiLSTMEncoder
from src.model.module.linear_crf_inferencer import LinearCRF
from src.model.module.linear_encoder import LinearEncoder
from src.model.embedder import TransformersEmbedder
from typing import Tuple, Union

from src.data.data_utils import START_TAG, STOP_TAG, PAD


class TransformersCRF(nn.Module):
    def __init__(self, config):
        super(TransformersCRF, self).__init__()
        self.transformer = TransformersEmbedder(
            transformer_model_name=config.embedder_type
        )
        if config.hidden_dim > 0:
            self.encoder = BiLSTMEncoder(
                label_size=config.label_size,
                input_dim=self.transformer.get_output_dim(),
                hidden_dim=config.hidden_dim,
                drop_lstm=config.dropout,
            )
        else:
            self.encoder = LinearEncoder(
                label_size=config.label_size,
                input_dim=self.transformer.get_output_dim(),
            )
        self.inferencer = LinearCRF(
            label_size=config.label_size,
            label2idx=config.label2idx,
            add_iobes_constraint=config.add_iobes_constraint,
            idx2labels=config.idx2labels,
        )
        self.pad_idx = config.label2idx[PAD]

    def forward(
        self,
        subword_input_ids: torch.Tensor,
        word_seq_lens: torch.Tensor,
        orig_to_tok_index: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        is_train: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Calculate the negative loglikelihood.
        :param subword_input_ids: (batch_size x max_seq_len)
        :param word_seq_lens: (batch_size) note: not subword
        :param context_emb: (batch_size x max_seq_len x context_emb_size)
        :param chars: (batch_size x max_seq_len x max_char_len)
        :param char_seq_lens: (batch_size x max_seq_len)
        :param labels: (batch_size x max_seq_len)
        :param is_train: whether to calculate the loss or not.
                        We use this for distributed training, so that we don't have to
                        add another function for `decode`
        :return: the total negative log-likelihood loss
        """
        word_rep = self.transformer(
            subword_input_ids, orig_to_tok_index, attention_mask
        )
        encoder_scores = self.encoder(word_rep, word_seq_lens)
        batch_size = word_rep.size(0)
        sent_len = word_rep.size(1)
        maskTemp = (
            torch.arange(1, sent_len + 1, dtype=torch.long, device=word_rep.device)
            .view(1, sent_len)
            .expand(batch_size, sent_len)
        )
        mask = torch.le(
            maskTemp, word_seq_lens.view(batch_size, 1).expand(batch_size, sent_len)
        )
        if is_train:
            unlabed_score, labeled_score = self.inferencer(
                encoder_scores, word_seq_lens, labels, mask
            )
            return unlabed_score - labeled_score
        else:
            bestScores, decodeIdx = self.inferencer.decode(
                encoder_scores, word_seq_lens
            )
            return bestScores, decodeIdx
