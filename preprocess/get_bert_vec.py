#
# @author: Allan
#
from typing import List, Tuple

from config.transformers_dataset import  Reader
import numpy as np
import pickle
import sys
from tqdm import tqdm
from bert_serving.client import BertClient
from transformers import BertTokenizer

def load_bert_client() -> BertClient:
    """
    Load a Bert Client
    :return:
    """
    return BertClient()

def bert_tokenize_words(tokenizer, words: List[str], mode: str) -> Tuple[List[str], List[int]]:
    word_piece_tokens = []
    orig_to_tok_index = []
    assert mode == "first" or mode == "last"
    for word in words:
        if mode == "first":
            orig_to_tok_index.append(len(word_piece_tokens))
        word_piece_tokens.extend(tokenizer.tokenize(word))
        if mode == "last":
            orig_to_tok_index.append(len(word_piece_tokens)-1)
    return word_piece_tokens, orig_to_tok_index

def read_parse_write(tokenizer: BertTokenizer, bert_client: BertClient, infile: str, outfile: str, mode) -> None:
    """
    Read the input files and write the vectors to the output files
    :param bert_client: BertClient
    :param infile: input files for the sentences
    :param outfile: output vector files
    :param mode: the mode of bert word piece
    :return:
    """
    reader = Reader()
    insts = reader.read_txt(infile, -1)
    f = open(outfile, 'wb')
    all_vecs = []
    all_sents = []
    for inst in insts:
        all_sents.append(inst.input.words)
    for sent in tqdm(all_sents, desc="BERT encoding"):
        word_piece_tokens, word_to_piece_index = bert_tokenize_words(tokenizer, sent, mode=mode)
        bert_vec =np.squeeze(bert_client.encode([word_piece_tokens], is_tokenized=True),
                             axis=0)[1:-1, :] ## exclude the [CLS] and [SEP]
        bert_vec = bert_vec[word_to_piece_index, :]
        print(bert_vec.shape)
        all_vecs.append(bert_vec)

    print("Finishing embedding BERT sequences, saving the vector files.")
    pickle.dump(all_vecs, f)
    f.close()


def get_vector(tokenizer, mode):

    bert_client = load_bert_client()
    dataset= sys.argv[1]


    # Read train
    file = "data/"+dataset+"/train.txt"
    outfile = file + ".bert.vec"
    read_parse_write(tokenizer, bert_client, file, outfile, mode)

    # Read dev
    file = "data/"+dataset+"/dev.txt"
    outfile = file + ".bert.vec"
    read_parse_write(tokenizer, bert_client, file, outfile, mode)

    # Read test
    file = "data/"+dataset+"/test.txt"
    outfile = file + ".bert.vec"
    read_parse_write(tokenizer, bert_client, file, outfile, mode)



if __name__ == "__main__":

    ### Remember to start the bert server first
    """
    # download the model files from https://github.com/hanxiao/bert-as-service
    
    bert-serving-start -model_dir ./model_files/uncased_L-12_H-768_A-12/ -num_worker=4 -cpu -pooling_strategy NONE -max_seq_len NONE -pooling_layer -1
    
    `pooling_strategy`  NONE means use last layer
    `max_seq_len` NONE means dynamically set the max length
    `pooling_layer` -1 means take the last layer
    """
    huggingface_bert_model = "bert-base-cased"
    mode = "last"  ## take the first wordpiece to represent the word.  "first" or "last"
    tokenizer = BertTokenizer.from_pretrained(huggingface_bert_model)
    get_vector(tokenizer, mode)
