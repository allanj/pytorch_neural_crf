import torch.nn as nn
import torch
from src.data.data_utils import START_TAG, STOP_TAG, PAD
from src.config.utils import log_sum_exp_pytorch
from typing import Dict, List
from typing import Tuple
from overrides import overrides
import math


class FastLinearCRF(nn.Module):


    def __init__(self, label_size:int, label2idx:Dict[str, int], add_iobes_constraint: bool = False,
                 idx2labels: List[str] = None):
        super(FastLinearCRF, self).__init__()

        self.label_size = label_size

        self.label2idx = label2idx
        self.idx2labels = idx2labels
        self.start_idx = self.label2idx[START_TAG]
        self.end_idx = self.label2idx[STOP_TAG]
        # self.pad_idx = self.label2idx[PAD]

        # initialize the following transition (anything never cannot -> start. end never  cannot-> anything. Same thing for the padding label)
        init_transition = torch.randn(self.label_size, self.label_size)
        init_transition[:, self.start_idx] = -10000.0
        init_transition[self.end_idx, :] = -10000.0
        init_transition[:, self.pad_idx] = -10000.0
        init_transition[self.pad_idx, :] = -10000.0
        if add_iobes_constraint:
            self.add_constraint_for_iobes(init_transition)

        self.transition = nn.Parameter(init_transition)

    def add_constraint_for_iobes(self, transition: torch.Tensor):
        print("[Info] Adding IOBES constraints")
        ## add constraint:
        for prev_label in self.idx2labels:
            if prev_label == START_TAG or prev_label == STOP_TAG or prev_label == PAD:
                continue
            for next_label in self.idx2labels:
                if next_label == START_TAG or next_label == STOP_TAG or next_label == PAD:
                    continue
                if prev_label == "O" and (next_label.startswith("I-") or next_label.startswith("E-")):
                    transition[self.label2idx[prev_label], self.label2idx[next_label]] = -10000.0
                if prev_label.startswith("B-") or prev_label.startswith("I-"):
                    if next_label.startswith("O") or next_label.startswith("B-") or next_label.startswith("S-"):
                        transition[self.label2idx[prev_label], self.label2idx[next_label]] = -10000.0
                    elif prev_label[2:] != next_label[2:]:
                        transition[self.label2idx[prev_label], self.label2idx[next_label]] = -10000.0
                if prev_label.startswith("S-") or prev_label.startswith("E-"):
                    if next_label.startswith("I-") or next_label.startswith("E-"):
                        transition[self.label2idx[prev_label], self.label2idx[next_label]] = -10000.0
        ##constraint for start and end
        for label in self.idx2labels:
            if label.startswith("I-") or label.startswith("E-"):
                transition[self.start_idx, self.label2idx[label]] = -10000.0
            if label.startswith("I-") or label.startswith("B-"):
                transition[self.label2idx[label], self.end_idx] = -10000.0

    @overrides
    def forward(self, lstm_scores, word_seq_lens, tags, mask):
        """
        Calculate the negative log-likelihood
        :param lstm_scores:
        :param word_seq_lens:
        :param tags:
        :param mask:
        :return:
        """
        all_scores=  self.calculate_all_scores(lstm_scores= lstm_scores)
        unlabed_score = self.forward_unlabeled(all_scores, word_seq_lens)
        labeled_score = self.forward_labeled(all_scores, word_seq_lens, tags, mask)
        return unlabed_score, labeled_score

    def forward_unlabeled(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Calculate the scores with the forward algorithm. Basically calculating the normalization term
        :param all_scores: (batch_size x max_seq_len x num_labels x num_labels) from (lstm scores + transition scores).
        :param word_seq_lens: (batch_size)
        :return: The score for all the possible structures.
        """
        batch_size = all_scores.size(0)
        seq_len = all_scores.size(1)
        dev_num = all_scores.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        depth = math.ceil(math.log2(seq_len))

        padded_length = int(math.pow(2, depth))
        sweep_score = torch.zeros(batch_size, padded_length, depth + 1, self.label_size, self.label_size, device=curr_dev)

        sweep_score[:, :seq_len, 0, :, :] = all_scores

        step_size = 2
        f_start = 0
        b_start = 1
        for d in range(depth):
            ##correct
            forward_score = sweep_score[:, f_start::step_size, d, :, :].unsqueeze(-1).expand(batch_size, padded_length // step_size, self.label_size, self.label_size, self.label_size)
            ##checking
            backward_score = sweep_score[:, b_start::step_size, d, :, :].unsqueeze(2).expand(batch_size, padded_length // step_size, self.label_size, self.label_size, self.label_size)
            sweep_score[:, b_start::step_size, d + 1, :, :] = torch.logsumexp(forward_score + backward_score, dim=-2)
            # sweep_score[:, b_start::step_size, d + 1, :, :] = torch.sum(forward_score + backward_score, dim=-2)
            f_start = b_start
            b_start = b_start + step_size
            step_size *= 2
        print(f"depth is {depth}, step_size: {step_size}")


        ##doing down_sweep
        step_size = step_size // 2
        sweep_score[:, -1, -1, :, :] = 0 # -float("Inf")
        # sweep_score[:, -1, -1, :, :] = 0 #for sum
        f_start = padded_length // 2 - 1
        b_start = padded_length - 1
        #log sum exp
        first_mask = torch.full([self.label_size, self.label_size, self.label_size], fill_value=-float("Inf"), device=curr_dev)
        # sum
        # first_mask = torch.full([self.label_size, self.label_size, self.label_size], fill_value=0, device=curr_dev)
        idxs = torch.arange(self.label_size, device=curr_dev)
        interleave_idxs = idxs.repeat_interleave(self.label_size)
        first_mask[interleave_idxs, interleave_idxs, idxs.repeat(self.label_size)] = 0  # log sum exp is 0. to pass over
        # first_mask[interleave_idxs, interleave_idxs, idxs.repeat(self.label_size)] = 1 # sum is 1. to pass through
        first_mask = first_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, self.label_size, self.label_size, self.label_size)
        ## for log sum exp
        zero_mask = torch.zeros(self.label_size, device=curr_dev).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
            batch_size, 1, self.label_size, self.label_size, self.label_size
        )
        ## for sum
        # zero_mask = torch.ones(self.label_size, device=curr_dev).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
        #     batch_size, 1, self.label_size, self.label_size, self.label_size
        # )
        for d in range(depth-1, -1, -1):
            length = padded_length // step_size
            ## backward score, calculate temporary
            temporary = sweep_score[:, f_start::step_size, d, :, :].clone()
            temporary =  temporary.unsqueeze(2).expand(batch_size, length, self.label_size, self.label_size, self.label_size)

            sweep_score[:, f_start::step_size, d, :, :] = sweep_score[:, b_start::step_size, d + 1, :, :].clone()

            ##forward_score
            forward_score = sweep_score[:, b_start::step_size, d + 1, :, :].unsqueeze(-1).expand(batch_size, length, self.label_size, self.label_size, self.label_size)
            curr_zero_mask = zero_mask.expand(batch_size, length - 1, self.label_size, self.label_size, self.label_size)
            mask = torch.cat([first_mask, curr_zero_mask], dim=1)
            # calculate backward originate score
            sweep_score[:, b_start::step_size, d, :, :] = torch.logsumexp(forward_score + temporary + mask, dim=-2)
            # sweep_score[:, b_start::step_size, d, :, :] = torch.sum(forward_score + mask * temporary, dim=-2) #

            b_start = f_start
            step_size = step_size // 2
            f_start = f_start - step_size // 2


        curr_zero_mask = zero_mask.expand(batch_size, seq_len - 1, self.label_size, self.label_size, self.label_size)
        mask = torch.cat([first_mask, curr_zero_mask], dim=1)
        sweep_score[:, :seq_len, 0, :, :] = torch.logsumexp(
            sweep_score[:, :seq_len, 0, :, :].unsqueeze(-1).expand(batch_size, seq_len, self.label_size, self.label_size, self.label_size) +
            all_scores.unsqueeze(2).expand(batch_size, seq_len, self.label_size, self.label_size, self.label_size) + mask,
            dim= -2
        )
        # sweep_score[:, :seq_len, 0, :, :] = torch.sum(
        #     sweep_score[:, :seq_len, 0, :, :].unsqueeze(-1).expand(batch_size, seq_len, self.label_size, self.label_size, self.label_size) +
        #     all_scores.unsqueeze(2).expand(batch_size, seq_len, self.label_size, self.label_size, self.label_size) * mask,
        #     dim=-2
        # )

        ### batch_size x label_size
        last_alpha = torch.gather(sweep_score[:, :, 0, self.start_idx, :], 1, word_seq_lens.view(batch_size, 1, 1).expand(batch_size, 1, self.label_size)-1).view(batch_size, self.label_size)
        last_alpha += self.transition[:, self.end_idx].view(1, self.label_size).expand(batch_size, self.label_size)
        last_alpha = log_sum_exp_pytorch(last_alpha.view(batch_size, self.label_size, 1)).view(batch_size)

        ## final score for the unlabeled network in this batch, with size: 1
        return torch.sum(last_alpha)

    def backward(self, lstm_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Backward algorithm. A benchmark implementation which is ready to use.
        :param lstm_scores: shape: (batch_size, sent_len, label_size) NOTE: the score from LSTMs, not `all_scores` (which add up the transtiion)
        :param word_seq_lens: shape: (batch_size,)
        :return: Backward variable
        """
        batch_size = lstm_scores.size(0)
        seq_len = lstm_scores.size(1)
        dev_num = lstm_scores.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        beta = torch.zeros(batch_size, seq_len, self.label_size, device=curr_dev)

        ## reverse the view of computing the score. we look from behind
        rev_score = self.transition.transpose(0, 1).view(1, 1, self.label_size, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size) + \
                    lstm_scores.view(batch_size, seq_len, 1, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size)

        ## The code below, reverse the score from [0 -> length]  to [length -> 0].  (NOTE: we need to avoid reversing the padding)
        perm_idx = torch.zeros(batch_size, seq_len, device=curr_dev)
        for batch_idx in range(batch_size):
            perm_idx[batch_idx][:word_seq_lens[batch_idx]] = torch.range(word_seq_lens[batch_idx] - 1, 0, -1)
        perm_idx = perm_idx.long()
        for i, length in enumerate(word_seq_lens):
            rev_score[i, :length] = rev_score[i, :length][perm_idx[i, :length]]

        ## backward operation
        beta[:, 0, :] = rev_score[:, 0, self.end_idx, :]
        for word_idx in range(1, seq_len):
            before_log_sum_exp = beta[:, word_idx - 1, :].view(batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size) + rev_score[:, word_idx, :, :]
            beta[:, word_idx, :] = log_sum_exp_pytorch(before_log_sum_exp)

        ## Following code is used to check the backward beta implementation
        last_beta = torch.gather(beta, 1, word_seq_lens.view(batch_size, 1, 1).expand(batch_size, 1, self.label_size) - 1).view(batch_size, self.label_size)
        last_beta += self.transition.transpose(0, 1)[:, self.start_idx].view(1, self.label_size).expand(batch_size, self.label_size)
        last_beta = log_sum_exp_pytorch(last_beta.view(batch_size, self.label_size, 1)).view(batch_size)

        # This part if optionally, if you only use `last_beta`.
        # Otherwise, you need this to reverse back if you also need to use beta
        for i, length in enumerate(word_seq_lens):
            beta[i, :length] = beta[i, :length][perm_idx[i, :length]]

        return torch.sum(last_beta)

    def forward_labeled(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor, tags: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        '''
        Calculate the scores for the gold instances.
        :param all_scores: (batch, seq_len, label_size, label_size)
        :param word_seq_lens: (batch, seq_len)
        :param tags: (batch, seq_len)
        :param masks: batch, seq_len
        :return: sum of score for the gold sequences Shape: (batch_size)
        '''
        batchSize = all_scores.shape[0]
        sentLength = all_scores.shape[1]

        ## all the scores to current labels: batch, seq_len, all_from_label?
        currentTagScores = torch.gather(all_scores, 3, tags.view(batchSize, sentLength, 1, 1).expand(batchSize, sentLength, self.label_size, 1)).view(batchSize, -1, self.label_size)
        if sentLength != 1:
            tagTransScoresMiddle = torch.gather(currentTagScores[:, 1:, :], 2, tags[:, : sentLength - 1].view(batchSize, sentLength - 1, 1)).view(batchSize, -1)
        tagTransScoresBegin = currentTagScores[:, 0, self.start_idx]
        endTagIds = torch.gather(tags, 1, word_seq_lens.view(batchSize, 1) - 1)
        tagTransScoresEnd = torch.gather(self.transition[:, self.end_idx].view(1, self.label_size).expand(batchSize, self.label_size), 1,  endTagIds).view(batchSize)
        score = torch.sum(tagTransScoresBegin) + torch.sum(tagTransScoresEnd)
        if sentLength != 1:
            score += torch.sum(tagTransScoresMiddle.masked_select(masks[:, 1:]))
        return score

    def calculate_all_scores(self, lstm_scores: torch.Tensor) -> torch.Tensor:
        """
        Calculate all scores by adding up the transition scores and emissions (from lstm).
        Basically, compute the scores for each edges between labels at adjacent positions.
        This score is later be used for forward-backward inference
        :param lstm_scores: emission scores.
        :return: (batch_size, sentence_len, label size, label size)
           start -> 0
           0 -> 1
           n-1 -> n
           we don't have the score from (n -> end) because we don't have emission at "end" tag
        """
        batch_size = lstm_scores.size(0)
        seq_len = lstm_scores.size(1)
        scores = self.transition.view(1, 1, self.label_size, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size) + \
                 lstm_scores.view(batch_size, seq_len, 1, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size)
        return scores

    def decode(self, features, wordSeqLengths) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        :param batchInput:
        :return:
        """
        all_scores = self.calculate_all_scores(features)
        bestScores, decodeIdx = self.viterbi_decode(all_scores, wordSeqLengths)
        return bestScores, decodeIdx

    def viterbi_decode(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use viterbi to decode the instances given the scores and transition parameters
        :param all_scores: (batch_size x max_seq_len x num_labels)
        :param word_seq_lens: (batch_size)
        :return: the best scores as well as the predicted label ids.
               (batch_size) and (batch_size x max_seq_len)
        """
        batchSize = all_scores.shape[0]
        sentLength = all_scores.shape[1]
        dev_num = all_scores.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        scoresRecord = torch.zeros([batchSize, sentLength, self.label_size], device=curr_dev)
        idxRecord = torch.zeros([batchSize, sentLength, self.label_size], dtype=torch.int64, device=curr_dev)
        mask = torch.ones_like(word_seq_lens, dtype=torch.int64, device=curr_dev)
        startIds = torch.full((batchSize, self.label_size), self.start_idx, dtype=torch.int64, device=curr_dev)
        decodeIdx = torch.LongTensor(batchSize, sentLength).to(curr_dev)

        scores = all_scores
        # scoresRecord[:, 0, :] = self.getInitAlphaWithBatchSize(batchSize).view(batchSize, self.label_size)
        scoresRecord[:, 0, :] = scores[:, 0, self.start_idx, :]  ## represent the best current score from the start, is the best
        idxRecord[:,  0, :] = startIds
        for wordIdx in range(1, sentLength):
            ### scoresIdx: batch x from_label x to_label at current index.
            scoresIdx = scoresRecord[:, wordIdx - 1, :].view(batchSize, self.label_size, 1).expand(batchSize, self.label_size,
                                                                                  self.label_size) + scores[:, wordIdx, :, :]
            idxRecord[:, wordIdx, :] = torch.argmax(scoresIdx, 1)  ## the best previous label idx to crrent labels
            scoresRecord[:, wordIdx, :] = torch.gather(scoresIdx, 1, idxRecord[:, wordIdx, :].view(batchSize, 1, self.label_size)).view(batchSize, self.label_size)

        lastScores = torch.gather(scoresRecord, 1, word_seq_lens.view(batchSize, 1, 1).expand(batchSize, 1, self.label_size) - 1).view(batchSize, self.label_size)  ##select position
        lastScores += self.transition[:, self.end_idx].view(1, self.label_size).expand(batchSize, self.label_size)
        decodeIdx[:, 0] = torch.argmax(lastScores, 1)
        bestScores = torch.gather(lastScores, 1, decodeIdx[:, 0].view(batchSize, 1))

        for distance2Last in range(sentLength - 1):
            lastNIdxRecord = torch.gather(idxRecord, 1, torch.where(word_seq_lens - distance2Last - 1 > 0, word_seq_lens - distance2Last - 1, mask).view(batchSize, 1, 1).expand(batchSize, 1, self.label_size)).view(batchSize, self.label_size)
            decodeIdx[:, distance2Last + 1] = torch.gather(lastNIdxRecord, 1, decodeIdx[:, distance2Last].view(batchSize, 1)).view(batchSize)

        return bestScores, decodeIdx


if __name__ == '__main__':
    import random
    seed = 42
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    import time
    # test fast crf
    labels = ['a', START_TAG, STOP_TAG]
    label2idx = {'a':0, START_TAG: 1, STOP_TAG: 2}
    model = FastLinearCRF(label_size=len(labels), label2idx=label2idx,
                          idx2labels=labels)
    seq_len = 129
    batch_size = 20
    all_lengths = torch.randint(1, seq_len, (batch_size,))
    print(all_lengths)
    all_scores = torch.randn(batch_size,  max(all_lengths), len(labels), len(labels))
    word_seq_lens = torch.LongTensor(all_lengths)
    start = time.time()
    output = model.forward_unlabeled(all_scores=all_scores, word_seq_lens=word_seq_lens)
    end = time.time()
    print(f"running time: {(end-start)*1000}")
    print(output)