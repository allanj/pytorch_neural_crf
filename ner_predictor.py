
import pickle

from model import NNCRF
import torch

from config import simple_batching, ContextEmb
from typing import List, Union, Tuple
from src.common import Instance, Sentence
import tarfile

from allennlp.commands.elmo import ElmoEmbedder
from preprocess.get_elmo_vec import load_elmo, parse_sentence


"""
Predictor usage example:

    sentence = "This is a sentence"
    # Or you can make a list of sentence:
    # sentence = ["This is a sentence", "This is the second sentence"]
    
    model_path = "english_model.tar.gz"
    model = NERPredictor(model_path)
    prediction = model.predict(sentence)
    print(prediction)

"""


class NERPredictor:
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


def read_parse_write(elmo: ElmoEmbedder, insts: List[Instance], mode: str = "average") -> None:
    """
    Attach the instances into the sentence/
    :param elmo: ELMo embedder
    :param insts: List of instance
    :param mode: the mode of elmo vectors
    :return:
    """
    for inst in insts:
        vec = parse_sentence(elmo, inst.input.words, mode=mode)
        inst.elmo_vec = vec
