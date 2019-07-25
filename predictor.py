
import pickle

from model import NNCRF
import torch

from config import Config, simple_batching, batching_list_instances
from typing import List
from common import Instance, Sentence
import numpy as np

def predict_insts(config: Config, model: NNCRF, batch_insts_ids) -> List[List[str]]:
    batch_max_scores, batch_max_ids = model.decode(batch_insts_ids)
    predictions = []
    for idx in range(len(batch_max_ids)):
        length = batch_insts_ids[1][idx]
        prediction = batch_max_ids[idx][:length].tolist()
        prediction = prediction[::-1]
        prediction = [config.idx2labels[l] for l in prediction]
        predictions.append(prediction)
    return predictions

def sent_to_insts(sentence: str) -> List[Instance]:
    words = sentence.split()
    return[Instance(Sentence(words))]

def sents_to_insts(sentences: List[str]) -> List[Instance]:
    insts = []
    for sentence in sentences:
        words = sentence.split()
        insts.append(Instance(Sentence(words)))
    return insts


def create_batch_data(config: Config, insts: List[Instance]):
    return simple_batching(config, insts)




def load_model(conf_file, model_file):

    f = open(conf_file, 'rb')
    conf = pickle.load(f)  # variables come out in the order you put them in
    # default batch size for conf is `10`
    f.close()
    model = NNCRF(conf, print_info=False)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    return model, conf


def predict(conf, model, sentences):

    insts = sents_to_insts(sentences)
    conf.map_insts_ids(insts)
    test_batches = create_batch_data(conf, insts)
    predictions = predict_insts(conf, model, test_batches)
    return predictions


sentence = ["EU rejects German", "EU rejects German call to boycott"]


conf_file = "model_files/config.m"
model_file = "model_files/lstm_200_crf_conll2003_500_none_context_sgd_lr_0.01.m"
model, conf = load_model(conf_file, model_file)

print(predict(conf, model, sentence))
