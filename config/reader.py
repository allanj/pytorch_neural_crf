# 
# @author: Allan
#

from tqdm import tqdm
from common.sentence import Sentence
from common.instance import Instance
from typing import List
import re
import pickle

class Reader:


    def __init__(self, digit2zero:bool=True):
        self.digit2zero = digit2zero
        self.vocab = set()

    def read_txt(self, file: str, number: int = -1, is_train: bool = True) -> List[Instance]:
        print("Reading file: " + file)
        insts = []
        # vocab = set() ## build the vocabulary
        with open(file, 'r', encoding='utf-8') as f:
            words = []
            labels = []
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line == "":
                    insts.append(Instance(Sentence(words), labels))
                    words = []
                    labels = []
                    if len(insts) == number:
                        break
                    continue
                word, label = line.split()
                if self.digit2zero:
                    word = re.sub('\d', '0', word) # replace digit with 0.
                words.append(word)
                self.vocab.add(word)
                labels.append(label)
        print("number of sentences: {}".format(len(insts)))
        return insts

    def load_elmo_vec(self, file, insts):
        f = open(file, 'rb')
        all_vecs = pickle.load(f)  # variables come out in the order you put them in
        f.close()
        size = 0
        for vec, inst in zip(all_vecs, insts):
            inst.elmo_vec = vec
            size = vec.shape[1]
            # print(str(vec.shape[0]) + ","+ str(len(inst.input.words)) + ", " + str(inst.input.words))
            assert(vec.shape[0] == len(inst.input.words))
        return size


