
import pickle

from model import NNCRF
import torch

from config import Config, simple_batching
from typing import List
from common import Instance, Sentence

def batching_list_instances(config: Config, insts: List[Instance]):
    train_num = len(insts)
    batch_size = config.batch_size
    total_batch = train_num // batch_size + 1 if train_num % batch_size != 0 else train_num // batch_size
    batched_data = []
    for batch_id in range(total_batch):
        one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        batched_data.append(simple_batching(config, one_batch_insts))

    return batched_data

def test_model(config: Config, test_insts):
    model_name = "model_files/lstm_{}_crf_{}_{}_{}_context_{}_lr_{}.m".format(config.hidden_dim, config.dataset,
                                                                              config.train_num, config.context_emb.name,
                                                                              config.optimizer.lower(),
                                                                              config.learning_rate)
    res_name = "results/lstm_{}_crf_{}_{}_{}_context_{}_lr_{}..results".format(config.hidden_dim, config.dataset,
                                                                               config.train_num,
                                                                               config.context_emb.name,
                                                                               config.optimizer.lower(),
                                                                               config.learning_rate)

    model = NNCRF(config)
    model.load_state_dict(torch.load(model_name))
    model.eval()

    evaluate_model(config, model, test_batches, "test", test_insts)
    write_results(res_name, test_insts)


def sent_to_insts(sentence: str) -> List[Instance]:
    words = sentence.split()
    return[Instance(Sentence(words))]


sentence = "I like Singapore"


conf_file = "model_files/config.m"
model_file = "model_files/lstm_200_crf_conll2003_500_none_context_sgd_lr_0.01.m"
f = open(conf_file, 'rb')
conf = pickle.load(f)  # variables come out in the order you put them in
f.close()

model = NNCRF(conf)
model.load_state_dict(torch.load(model_file))
model.eval()

test_batches = batching_list_instances(conf, test_insts)


