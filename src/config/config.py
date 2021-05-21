# 
# @author: Allan
#

import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Union, Any
# from src.common import Instance
import torch
from enum import Enum
import os

from termcolor import colored



class ContextEmb(Enum):
    none = 0
    elmo = 1
    bert = 2 # not support yet
    flair = 3 # not support yet


class Config:
    def __init__(self, args) -> None:
        """
        Construct the arguments and some hyperparameters
        :param args:
        """

        # Model hyper parameters
        self.embedding_file = args.embedding_file if "embedding_file" in args.__dict__ else None
        self.embedding_dim = args.embedding_dim if "embedding_dim" in args.__dict__ else None
        self.static_context_emb = ContextEmb[args.static_context_emb] if "static_context_emb" in args.__dict__ else ContextEmb.none
        self.context_emb_size = 0
        self.embedding, self.embedding_dim = self.read_pretrain_embedding() if "embedding_file" in args.__dict__ else (None, None)
        self.word_embedding = None
        self.seed = args.seed
        self.hidden_dim = args.hidden_dim
        self.use_brnn = True
        self.num_layers = 1
        self.dropout = args.dropout
        self.char_emb_size = 25
        self.charlstm_hidden_dim = 50
        self.use_char_rnn = args.use_char_rnn if "use_char_rnn" in args.__dict__ else None

        self.embedder_type = args.embedder_type if "embedder_type" in args.__dict__ else None
        self.parallel_embedder = args.parallel_embedder if "parallel_embedder" in args.__dict__ else None
        self.add_iobes_constraint = args.add_iobes_constraint

        # Data specification
        self.dataset = args.dataset
        self.train_file = "data/" + self.dataset + "/train.txt"
        self.dev_file = "data/" + self.dataset + "/dev.txt"
        self.test_file = "data/" + self.dataset + "/test.txt"
        self.train_num = args.train_num
        self.dev_num = args.dev_num
        self.test_num = args.test_num

        # Training hyperparameter
        self.model_folder = args.model_folder
        self.optimizer = args.optimizer.lower()
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum if "momentum" in args.__dict__ else None
        self.l2 = args.l2
        self.num_epochs = args.num_epochs
        self.use_dev = True
        self.batch_size = args.batch_size
        self.clip = 5
        self.lr_decay = args.lr_decay
        self.device = torch.device(args.device)
        self.max_no_incre = args.max_no_incre
        self.max_grad_norm = args.max_grad_norm if "max_grad_norm" in args.__dict__ else None

        self.print_detail_f1 = args.print_detail_f1
        self.earlystop_atr = args.earlystop_atr

    def read_pretrain_embedding(self) -> Tuple[Union[Dict[str, np.array], None], int]:
        """
        Read the pretrained word embeddings, return the complete embeddings and the embedding dimension
        :return:
        """
        print("reading the pretraing embedding: %s" % (self.embedding_file))
        if self.embedding_file is None:
            print("pretrain embedding in None, using random embedding")
            return None, self.embedding_dim
        else:
            exists = os.path.isfile(self.embedding_file)
            if not exists:
                print(colored("[Warning] pretrain embedding file not exists, using random embedding",  'red'))
                return None, self.embedding_dim
                # raise FileNotFoundError("The embedding file does not exists")
        embedding_dim = -1
        embedding = dict()
        with open(self.embedding_file, 'r', encoding='utf-8') as file:
            for line in tqdm(file.readlines()):
                line = line.strip()
                if len(line) == 0:
                    continue
                tokens = line.split()
                if embedding_dim < 0:
                    embedding_dim = len(tokens) - 1
                else:
                    # print(tokens)
                    # print(embedding_dim)
                    assert (embedding_dim + 1 == len(tokens))
                embedd = np.empty([1, embedding_dim])
                embedd[:] = tokens[1:]
                first_col = tokens[0]
                embedding[first_col] = embedd
        return embedding, embedding_dim



    def build_emb_table(self, word2idx: Dict[str, int]) -> None:
        """
        build the embedding table with pretrained word embeddings (if given otherwise, use random embeddings)
        :return:
        """
        print("Building the embedding table for vocabulary...")
        scale = np.sqrt(3.0 / self.embedding_dim)
        if self.embedding is not None:
            print("[Info] Use the pretrained word embedding to initialize: %d x %d" % (len(word2idx), self.embedding_dim))
            self.word_embedding = np.empty([len(word2idx), self.embedding_dim])
            for word in word2idx:
                if word in self.embedding:
                    self.word_embedding[word2idx[word], :] = self.embedding[word]
                elif word.lower() in self.embedding:
                    self.word_embedding[word2idx[word], :] = self.embedding[word.lower()]
                else:
                    # self.word_embedding[self.word2idx[word], :] = self.embedding[self.UNK]
                    self.word_embedding[word2idx[word], :] = np.random.uniform(-scale, scale, [1, self.embedding_dim])
            self.embedding = None  ## remove the pretrained embedding to save memory.
        else:
            self.word_embedding = np.empty([len(word2idx), self.embedding_dim])
            for word in word2idx:
                self.word_embedding[word2idx[word], :] = np.random.uniform(-scale, scale, [1, self.embedding_dim])

