
import pickle

from model import NNCRF
import torch

from config import Config, simple_batching, batching_list_instances, ContextEmb
from typing import List, Union, Tuple
from common import Instance, Sentence
import tarfile
import numpy as np

from allennlp.commands.elmo import ElmoEmbedder
def load_elmo(cuda_device: int) -> ElmoEmbedder:
    """
    Load a ElMo embedder
    :param cuda_device:
    :return:
    """
    return ElmoEmbedder(cuda_device=cuda_device)


def parse_sentence(elmo: ElmoEmbedder, words: List[str], mode:str="average") -> np.array:
    """
    Load an ELMo embedder.
    :param elmo: the ELMo model embedder, allows us to embed a sequence of words
    :param words: the input word tokens.
    :param mode:
    :return:
    """
    vectors = elmo.embed_sentence(words)
    if mode == "average":
        return np.average(vectors, 0)
    elif mode == 'weighted_average':
        return np.swapaxes(vectors, 0, 1)
    elif mode == 'last':
        return vectors[-1, :, :]
    elif mode == 'all':
        return vectors
    else:
        return vectors

def read_parse_write(elmo: ElmoEmbedder, insts, mode: str = "average") -> None:
    """
    Read the input files and write the vectors to the output files
    :param elmo: ELMo embedder
    :param infile: input files for the sentences
    :param outfile: output vector files
    :param mode: the mode of elmo vectors
    :return:
    """
    for inst in insts:
        vec = parse_sentence(elmo, inst.input.words, mode=mode)
        inst.elmo_vec = vec

class Predictor:
    """
    Usage:
    sentence = "This is a sentence"
    model_path = "model_files.tar.gz"
    model = Predictor(model_path)
    prediction = model.predict(sentence)
    """

    def __init__(self, model_archived_file:str, cuda_device: str = "cpu"):

        tar = tarfile.open(model_archived_file)
        tar.extractall()
        folder_name = tar.getnames()[0]
        tar.close()

        f = open(folder_name + "/config.conf", 'rb')
        self.conf = pickle.load(f)  # variables come out in the order you put them in
        # default batch size for conf is `10`
        f.close()
        device = torch.device(cuda_device)
        self.conf.device = device
        self.model = NNCRF(self.conf, print_info=False)
        self.model.load_state_dict(torch.load(folder_name + "/lstm_crf.m", map_location = device))
        self.model.eval()

        if self.conf.context_emb != ContextEmb.none:
            if cuda_device == "cpu":
                cuda_device = -1
            else:
                cuda_device = int(cuda_device.split(":")[1])
            self.elmo = load_elmo(cuda_device)

    def predict_insts(self, batch_insts_ids: Tuple) -> List[List[str]]:
        batch_max_scores, batch_max_ids = self.model.decode(batch_insts_ids)
        predictions = []
        for idx in range(len(batch_max_ids)):
            length = batch_insts_ids[1][idx]
            prediction = batch_max_ids[idx][:length].tolist()
            prediction = prediction[::-1]
            prediction = [self.conf.idx2labels[l] for l in prediction]
            predictions.append(prediction)
        return predictions

    def sent_to_insts(self, sentence: str) -> List[Instance]:
        words = sentence.split()
        return[Instance(Sentence(words))]

    def sents_to_insts(self, sentences: List[str]) -> List[Instance]:
        insts = []
        for sentence in sentences:
            words = sentence.split()
            insts.append(Instance(Sentence(words)))
        return insts

    def create_batch_data(self, insts: List[Instance]):
        return simple_batching(self.conf, insts)

    def predict(self, sentences: Union[str, List[str]]):

        sents = [sentences] if isinstance(sentences, str) else sentences
        insts = self.sents_to_insts(sents)
        self.conf.map_insts_ids(insts)
        if self.conf.context_emb != ContextEmb.none:
            read_parse_write(self.elmo, insts)
        test_batches = self.create_batch_data(insts)
        predictions = self.predict_insts(test_batches)
        if len(predictions) == 1:
            return predictions[0]
        else:
            return predictions



# sentence = "This is a sentence"
# model_path = "english_model.tar.gz"
# model = Predictor(model_path)
# prediction = model.predict(sentence)
# print(prediction)