#
# @author: Allan
#
from typing import List

from config.transformers_dataset import  Reader
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
import pickle
import sys
from tqdm import tqdm

def parse_sentence(elmo_vecs, mode:str="average") -> np.array:
    """
    Load an ELMo embedder.
    :param elmo_vecs: the ELMo model results for a single sentence
    :param mode:
    :return:
    """
    if mode == "average":
        return np.average(elmo_vecs, 0)
    elif mode == 'weighted_average':
        return np.swapaxes(elmo_vecs, 0, 1)
    elif mode == 'last':
        return elmo_vecs[-1, :, :]
    elif mode == 'all':
        return elmo_vecs
    else:
        return elmo_vecs


def load_elmo(cuda_device: int) -> ElmoEmbedder:
    """
    Load a ElMo embedder
    :param cuda_device:
    :return:
    """
    return ElmoEmbedder(cuda_device=cuda_device)


def read_parse_write(elmo: ElmoEmbedder, infile: str, outfile: str, mode: str = "average", batch_size=0) -> None:
    """
    Read the input files and write the vectors to the output files
    :param elmo: ELMo embedder
    :param infile: input files for the sentences
    :param outfile: output vector files
    :param mode: the mode of elmo vectors
    :return:
    """
    reader = Reader()
    insts = reader.read_txt(infile, -1)
    f = open(outfile, 'wb')
    all_vecs = []
    all_sents = []
    for inst in insts:
        all_sents.append(inst.input.words)
    if batch_size < 1: # Not using batch
        for sent in tqdm(all_sents, desc="Elmo Embedding"):        
            elmo_vecs = elmo.embed_sentence(sent) 
            vec = parse_sentence(elmo_vecs, mode=mode)    
            all_vecs.append(vec)
    else:   # Batched prediction
        for elmo_vecs in tqdm(elmo.embed_sentences(all_sents, batch_size=batch_size), desc="Elmo Embedding", total=len(all_sents)):
            vec = parse_sentence(elmo_vecs, mode=mode)
            all_vecs.append(vec)

    print("Finishing embedding ELMo sequences, saving the vector files.")
    pickle.dump(all_vecs, f)
    f.close()


def get_vector():

    cuda_device = 0 # >=0 for gpu, using GPU should be much faster.  < 0 for cpu.
    elmo = load_elmo(cuda_device)
    mode= "average"
    dataset=sys.argv[1]
    batch_size = 64 # >=1 for using batch-based inference


    # Read train
    file = "data/"+dataset+"/train.txt"
    outfile = file + ".elmo.vec"
    read_parse_write(elmo, file, outfile, mode, batch_size)

    # Read dev
    file = "data/"+dataset+"/dev.txt"
    outfile = file + ".elmo.vec"
    read_parse_write(elmo, file, outfile, mode, batch_size)

    # Read test
    file = "data/"+dataset+"/test.txt"
    outfile = file + ".elmo.vec"
    read_parse_write(elmo, file, outfile, mode, batch_size)



if __name__ == "__main__":
    get_vector()
