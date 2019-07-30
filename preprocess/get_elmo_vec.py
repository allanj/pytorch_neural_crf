#
# @author: Allan
#
from typing import List

from config.reader import  Reader
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
import pickle


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


def load_elmo(cuda_device: int) -> ElmoEmbedder:
    """
    Load a ElMo embedder
    :param cuda_device:
    :return:
    """
    return ElmoEmbedder(cuda_device=cuda_device)


def read_parse_write(elmo: ElmoEmbedder, infile: str, outfile: str, mode: str = "average") -> None:
    """
    Read the input files and write the vectors to the output files
    :param elmo: ELMo embedder
    :param infile: input files for the sentences
    :param outfile: output vector files
    :param mode: the mode of elmo vectors
    :return:
    """
    reader = Reader()
    insts = reader.read_txt(infile, -1, True)
    f = open(outfile, 'wb')
    all_vecs = []
    for inst in insts:
        vec = parse_sentence(elmo, inst.input.words, mode=mode)
        all_vecs.append(vec)
    print("Finishing embedding ELMo sequences, saving the vector files.")
    pickle.dump(all_vecs, f)
    f.close()


def get_vector():

    cuda_device = -1 # >=0 for gpu, using GPU should be much faster.
    elmo = load_elmo(cuda_device)
    mode= "average"
    dataset="conll2003"


    # Read train
    file = "../data/"+dataset+"/train.txt"
    outfile = file + ".elmo.vec"
    read_parse_write(elmo, file, outfile, mode)

    # Read dev
    file = "../data/"+dataset+"/dev.txt"
    outfile = file + ".elmo.vec"
    read_parse_write(elmo, file, outfile, mode)

    # Read test
    file = "../data/"+dataset+"/test.txt"
    outfile = file + ".elmo.vec"
    read_parse_write(elmo, file, outfile, mode)



if __name__ == "__main__":
    get_vector()