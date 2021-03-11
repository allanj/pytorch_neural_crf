import torch.nn as nn
import torch
from src.data.data_utils import START_TAG, STOP_TAG, PAD
from src.config.utils import log_sum_exp_pytorch
from typing import Dict, List
from typing import Tuple
import math
from src.model.module.linear_crf_inferencer import LinearCRF

class FastLinearCRF(LinearCRF):


    def __init__(self, label_size:int, label2idx:Dict[str, int], add_iobes_constraint: bool = False,
                 idx2labels: List[str] = None):
        super(FastLinearCRF, self).__init__(label_size=label_size,
                                            label2idx=label2idx,
                                            add_iobes_constraint=add_iobes_constraint,
                                            idx2labels=idx2labels)
        print("[WARNING] YOU ARE USING Log N CRF. Now, you do not need to reverse back the sequence after decoding. ")

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
        # print(f"depth is {depth}, step_size: {step_size}")


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

    def viterbi_decode(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

        argmax_idxs = torch.zeros(batch_size, padded_length, depth + 1, self.label_size, self.label_size, padded_length, device=curr_dev, dtype=torch.long)

        sweep_score[:, :seq_len, 0, :, :] = all_scores

        step_size = 2
        f_start = 0
        b_start = 1
        for d in range(depth):
            ##correct
            forward_score = sweep_score[:, f_start::step_size, d, :, :].unsqueeze(-1).expand(batch_size, padded_length // step_size, self.label_size, self.label_size, self.label_size)
            ##checking
            backward_score = sweep_score[:, b_start::step_size, d, :, :].unsqueeze(2).expand(batch_size, padded_length // step_size, self.label_size, self.label_size, self.label_size)
            # batch_size, length/step_size/2, label_size, label_size
            sweep_score[:, b_start::step_size, d + 1, :, :], current_argmax_idxs = torch.max(forward_score + backward_score, dim=-2)
            idxs = torch.arange(b_start, padded_length, step_size, device=curr_dev)
            position_idxs = torch.arange(f_start+1, padded_length, step_size, device=curr_dev)
            argmax_idxs[:, idxs, d + 1, :, :, position_idxs] = current_argmax_idxs.transpose(0,1)
            if d != 0:
                left_argmax_idxs = torch.gather(argmax_idxs[:, f_start::step_size, d,:, :, :], 3, current_argmax_idxs.unsqueeze(-1).expand_as(argmax_idxs[:, f_start::step_size, d,:, :, :]))
                right_argmax_idxs = torch.gather(argmax_idxs[:, b_start::step_size, d,:, :, :], 2, current_argmax_idxs.unsqueeze(-1).expand_as(argmax_idxs[:, f_start::step_size, d,:, :, :]))
                argmax_idxs[:, b_start::step_size, d + 1,:, :, :] += left_argmax_idxs
                argmax_idxs[:, b_start::step_size, d + 1, :, :, :] += right_argmax_idxs
            f_start = b_start
            b_start = b_start + step_size
            step_size *= 2
        # print(f"depth is {depth}, step_size: {step_size}")
        ## final argmax_idxs is valid from [1:] and concat a final best index to end..

        ##doing down_sweep
        step_size = step_size // 2
        sweep_score[:, -1, -1, :, :] = 0 # -float("Inf")
        argmax_idxs[:, -1, -1, :, :, :] = 0
        f_start = padded_length // 2 - 1
        b_start = padded_length - 1
        position_start = 0
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
        for d in range(depth-1, -1, -1):
            length = padded_length // step_size
            ## backward score, calculate temporary
            temporary = sweep_score[:, f_start::step_size, d, :, :].clone()
            temporary =  temporary.unsqueeze(2).expand(batch_size, length, self.label_size, self.label_size, self.label_size)

            temp_argmax = argmax_idxs[:, f_start::step_size, d, :, :, :].clone()
            argmax_idxs[:, f_start::step_size, d, :, :, :] = argmax_idxs[:, b_start::step_size, d + 1, :, :, :].clone()

            sweep_score[:, f_start::step_size, d, :, :] = sweep_score[:, b_start::step_size, d + 1, :, :].clone()

            ##forward_score
            forward_score = sweep_score[:, b_start::step_size, d + 1, :, :].unsqueeze(-1).expand(batch_size, length, self.label_size, self.label_size, self.label_size)
            curr_zero_mask = zero_mask.expand(batch_size, length - 1, self.label_size, self.label_size, self.label_size)
            mask = torch.cat([first_mask, curr_zero_mask], dim=1)
            # calculate backward originate score
            sweep_score[:, b_start::step_size, d, :, :], current_argmax_idxs = torch.max(forward_score + temporary + mask, dim=-2)
            # sweep_score[:, b_start::step_size, d, :, :] = torch.sum(forward_score + mask * temporary, dim=-2) #

            ##max score
            idxs = torch.arange(b_start, padded_length, step_size, device=curr_dev)
            position_idxs = torch.arange(position_start, padded_length, step_size, device=curr_dev)

            left_argmax_idxs = torch.gather(argmax_idxs[:, b_start::step_size, d + 1, :, :, :], 3, current_argmax_idxs.unsqueeze(-1).expand_as(argmax_idxs[:, b_start::step_size, d, :, :, :]))
            right_argmax_idxs = torch.gather(temp_argmax, 2, current_argmax_idxs.unsqueeze(-1).expand_as(temp_argmax))
            argmax_idxs[:, b_start::step_size, d, :, :, :] = right_argmax_idxs

            argmax_idxs[:, b_start::step_size, d, :, :, :] += left_argmax_idxs

            argmax_idxs[:, idxs, d, :, :, position_idxs] += current_argmax_idxs.transpose(0, 1)

            b_start = f_start
            step_size = step_size // 2
            f_start = f_start - step_size // 2

        curr_zero_mask = zero_mask.expand(batch_size, seq_len - 1, self.label_size, self.label_size, self.label_size)
        mask = torch.cat([first_mask, curr_zero_mask], dim=1)
        sweep_score[:, :seq_len, 0, :, :], current_argmax_idxs = torch.max(
            sweep_score[:, :seq_len, 0, :, :].unsqueeze(-1).expand(batch_size, seq_len, self.label_size, self.label_size, self.label_size) +
            all_scores.unsqueeze(2).expand(batch_size, seq_len, self.label_size, self.label_size, self.label_size) + mask,
            dim= -2
        )
        left_argmax_idxs = torch.gather(argmax_idxs[:, :seq_len, 0, :, :, :], 3, current_argmax_idxs.unsqueeze(-1).expand_as(argmax_idxs[:, :seq_len, 0, :, :, :]))

        argmax_idxs[:, :seq_len, 0, :, :, :] = left_argmax_idxs
        idxs = torch.arange(b_start, seq_len, step_size, device=curr_dev)
        position_idxs = torch.arange(position_start, seq_len, step_size, device=curr_dev)
        argmax_idxs[:, idxs, 0, :, :, position_idxs] += current_argmax_idxs.transpose(0, 1)
        # sweep_score[:, :seq_len, 0, :, :] = torch.sum(
        #     sweep_score[:, :seq_len, 0, :, :].unsqueeze(-1).expand(batch_size, seq_len, self.label_size, self.label_size, self.label_size) +
        #     all_scores.unsqueeze(2).expand(batch_size, seq_len, self.label_size, self.label_size, self.label_size) * mask,
        #     dim=-2
        # )

        final_argmax_indices = argmax_idxs[:, :, 0, self.start_idx, :, :] # batch_size x padded_length x self.to_label_size x indices

        ### batch_size x label_size
        last_alpha = torch.gather(sweep_score[:, :, 0, self.start_idx, :], 1, word_seq_lens.view(batch_size, 1, 1).expand(batch_size, 1, self.label_size)-1).view(batch_size, self.label_size)
        last_alpha += self.transition[:, self.end_idx].view(1, self.label_size).expand(batch_size, self.label_size)
        last_alpha, final_indices = torch.max(last_alpha, dim = -1) # final_indices: batch_size, best idx to end label.

        final_argmax_indices = torch.gather(final_argmax_indices, 1, word_seq_lens.view(batch_size, 1, 1, 1).expand(batch_size, 1, self.label_size, padded_length)-1).squeeze(1)
        b_idxs = torch.arange(0, batch_size, device=curr_dev)
        final_argmax_indices = final_argmax_indices[b_idxs,final_indices, :] # batch_size, padded_length
        final_argmax_indices = torch.cat([final_argmax_indices, torch.zeros(batch_size, 1, dtype=torch.long, device=curr_dev)], dim=-1)
        final_argmax_indices[b_idxs, word_seq_lens] = final_indices


        ## final score for the unlabeled network in this batch, with size: 1
        return last_alpha, final_argmax_indices[:, 1:(seq_len+1)]

if __name__ == '__main__':
    import random
    seed = 42
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    import time
    # test fast crf
    labels = ['a', PAD, START_TAG, STOP_TAG]
    label2idx = {'a': 0, PAD: 1, START_TAG: 2, STOP_TAG: 3}
    model = FastLinearCRF(label_size=len(labels), label2idx=label2idx,
                          idx2labels=labels)
    seq_len = 80
    batch_size = 5
    all_lengths = torch.randint(1, seq_len, (batch_size,))
    print(all_lengths)
    # all_lengths = torch.LongTensor([7, 14])
    all_scores = torch.randn(batch_size,  max(all_lengths), len(labels), len(labels))
    word_seq_lens = torch.LongTensor(all_lengths)
    start = time.time()
    output = model.forward_unlabeled(all_scores=all_scores, word_seq_lens=word_seq_lens)
    end = time.time()
    print(f"running time: {(end-start)*1000}")
    print(output)

    print("##testing decoding process.")
    with torch.no_grad():
        scores, indices = model.fast_viterbi(all_scores=all_scores, word_seq_lens=word_seq_lens)
    print(f"{scores}")
    # print(indices)
    for i, seq_len in enumerate(all_lengths):
        print(indices[i, :seq_len])