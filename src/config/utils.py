import torch
from typing import List, Tuple, Dict, Any
import pickle
import torch.optim as optim

import torch.nn as nn
from transformers import AdamW

from src.config import Config
from termcolor import colored
from src.data import Instance

def log_sum_exp_pytorch(vec: torch.Tensor) -> torch.Tensor:
    """
    Calculate the log_sum_exp trick for the tensor.
    :param vec: [batchSize * from_label * to_label].
    :return: [batchSize * to_label]
    """
    maxScores, idx = torch.max(vec, 1)
    maxScores[maxScores == -float("Inf")] = 0
    maxScoresExpanded = maxScores.view(vec.shape[0] ,1 , vec.shape[2]).expand(vec.shape[0], vec.shape[1], vec.shape[2])
    return maxScores + torch.log(torch.sum(torch.exp(vec - maxScoresExpanded), 1))

def lr_decay(config, optimizer: optim.Optimizer, epoch: int) -> optim.Optimizer:
    """
    Method to decay the learning rate
    :param config: configuration
    :param optimizer: optimizer
    :param epoch: epoch number
    :return:
    """
    lr = config.learning_rate / (1 + config.lr_decay * (epoch - 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('learning rate is set to: ', lr)
    return optimizer


def load_elmo_vec(file: str):
    """
    Load the elmo vectors and the vector will be saved within each instance with a member `elmo_vec`
    :param file: the vector files for the ELMo vectors
    :param insts: list of instances
    :return:
    """
    f = open(file, 'rb')
    all_vecs = pickle.load(f)  # variables come out in the order you put them in
    f.close()
    return all_vecs





def get_optimizer(config: Config, model: nn.Module,
                  weight_decay: float = 0.0,
                  eps: float = 1e-8,
                  warmup_step: int = 0):
    params = model.parameters()
    if config.optimizer.lower() == "sgd":
        print(colored("Using SGD: lr is: {}, L2 regularization is: {}".format(config.learning_rate, config.l2), 'yellow'))
        return optim.SGD(params, lr=config.learning_rate, weight_decay=float(config.l2))
    elif config.optimizer.lower() == "adam":
        print(colored(f"Using Adam, with learning rate: {config.learning_rate}", 'yellow'))
        return optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer.lower() == "adamw":
        print(colored(f"Using AdamW optimizeer with {config.learning_rate} learning rate, "
                      f"eps: {1e-8}", 'yellow'))
        return AdamW(model.parameters(), lr=config.learning_rate, eps=1e-8)
    else:
        print("Illegal optimizer: {}".format(config.optimizer))
        exit(1)



def write_results(filename: str, insts: List[Instance]):
    f = open(filename, 'w', encoding='utf-8')
    for inst in insts:
        for i in range(len(inst.words)):
            words = inst.ori_words
            output = inst.labels
            prediction = inst.prediction
            assert len(output) == len(prediction)
            f.write("{}\t{}\t{}\t{}\n".format(i, words[i], output[i], prediction[i]))
        f.write("\n")
    f.close()

def get_metric(p_num: int, total_num: int, total_predicted_num: int) -> Tuple[float, float, float]:
    """
    Return the metrics of precision, recall and f-score, based on the number
    (We make this small piece of function in order to reduce the code effort and less possible to have typo error)
    :param p_num:
    :param total_num:
    :param total_predicted_num:
    :return:
    """
    precision = p_num * 1.0 / total_predicted_num * 100 if total_predicted_num != 0 else 0
    recall = p_num * 1.0 / total_num * 100 if total_num != 0 else 0
    fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    return precision, recall, fscore


