import torch
import torch.nn as nn
from transformers import AutoModel
import logging

logger = logging.getLogger(__name__)

class TransformersEmbedder(nn.Module):
    """
    Encode the input with transformers model such as
    BERT, Roberta, and so on.
    """

    def __init__(self, transformer_model_name: str):
        super(TransformersEmbedder, self).__init__()
        output_hidden_states = False ## to use all hidden states or not
        logger.info(f"[Model Info] Loading pretrained language model {transformer_model_name}")

        self.model = AutoModel.from_pretrained(transformer_model_name,
                                               output_hidden_states= output_hidden_states,
                                               return_dict=True)
        """
        use the following line if you want to freeze the model, 
        but don't forget also exclude the parameters in the optimizer
        """
        # self.model.requires_grad = False

    def get_output_dim(self):
        ## use differnet model may have different attribute
        ## for example, if you are using GPT, it should be self.model.config.n_embd
        ## Check out https://huggingface.co/transformers/model_doc/gpt.html
        ## But you can directly write it as 768 as well.
        return self.model.config.hidden_size

    def forward(self, subword_input_ids: torch.Tensor,
                orig_to_token_index: torch.LongTensor,  ## batch_size * max_seq_leng
                attention_mask: torch.LongTensor) -> torch.Tensor:
        """

        :param subword_input_ids: (batch_size x max_wordpiece_len x hidden_size) the input id tensor
        :param orig_to_token_index: (batch_size x max_sent_len x hidden_size) the mapping from original word id map to subword token index
        :param attention_mask: (batch_size x max_wordpiece_len)
        :return:
        """
        subword_rep = self.model(**{"input_ids": subword_input_ids, "attention_mask": attention_mask}).last_hidden_state
        batch_size, _, rep_size = subword_rep.size()
        _, max_sent_len = orig_to_token_index.size()
        # select the word index.
        word_rep =  torch.gather(subword_rep[:, :, :], 1, orig_to_token_index.unsqueeze(-1).expand(batch_size, max_sent_len, rep_size))
        return word_rep