# 
# @author: Allan
#

from tqdm import tqdm
from typing import List, Dict
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
import collections
import numpy as np
from src.data.data_utils import convert_iobes, build_label_idx, check_all_labels_in_dict, check_all_obj_is_None

from src.data import Instance
from src.data.data_utils import UNK
import re

Feature = collections.namedtuple('Feature', 'words word_seq_len context_emb chars char_seq_lens labels')
Feature.__new__.__defaults__ = (None,) * 6


class NERDataset(Dataset):

    def __init__(self, file: str,
                 is_train: bool,
                 label2idx: Dict[str, int] = None,
                 number: int = -1):
        """
        Read the dataset into Instance
        """
        ## read all the instances. sentences and labels
        insts = self.read_txt(file=file, number=number)
        self.insts = insts
        if is_train:
            print(f"[Data Info] Using the training set to build label index")
            assert label2idx is None
            ## build label to index mapping. e.g., B-PER -> 0, I-PER -> 1
            idx2labels, label2idx = build_label_idx(insts)
            self.idx2labels = idx2labels
            self.label2idx = label2idx
        else:
            assert label2idx is not None ## for dev/test dataset we don't build label2idx, pass in label2idx argument
            self.label2idx = label2idx
            check_all_labels_in_dict(insts=insts, label2idx=self.label2idx)

    def convert_instances_to_feature_tensors(self, word2idx: Dict[str, int], char2idx: Dict[str, int], elmo_vecs: np.ndarray = None):
        self.inst_ids = []
        for i, inst in enumerate(self.insts):
            words = inst.words
            word_ids = []
            char_ids = []
            output_ids = []
            char_seq_lens = []
            for word in words:
                if word in word2idx:
                    word_ids.append(word2idx[word])
                else:
                    word_ids.append(word2idx[UNK])
                char_id = []
                char_seq_lens.append(len(word))
                for c in word:
                    if c in char2idx:
                        char_id.append(char2idx[c])
                    else:
                        char_id.append(char2idx[UNK])
                char_ids.append(char_id)
            if inst.labels is not None:
                for label in inst.labels:
                    output_ids.append(self.label2idx[label])
            context_emb = elmo_vecs[i] if elmo_vecs is not None else None
            self.inst_ids.append(Feature(words = word_ids,
                                         chars = char_ids,
                                         word_seq_len = len(words),
                                         char_seq_lens = char_seq_lens,
                                         context_emb = context_emb,
                                         labels = output_ids if inst.labels is not None else None))



    def read_txt(self, file: str, number: int = -1) -> List[Instance]:
        print(f"[Data Info] Reading file: {file}, labels will be converted to IOBES encoding")
        print(f"[Data Info] Modify src/data/ner_dataset.read_txt function if you have other requirements")
        insts = []
        with open(file, 'r', encoding='utf-8') as f:
            words = []
            ori_words = []
            labels = []
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line == "":
                    labels = convert_iobes(labels)
                    insts.append(Instance(words=words, ori_words=ori_words, labels=labels))
                    words = []
                    ori_words = []
                    labels = []
                    if len(insts) == number:
                        break
                    continue
                ls = line.split()
                word, label = ls[0],ls[-1]
                ori_words.append(word)
                word = re.sub('\d', '0', word)
                words.append(word)
                labels.append(label)
        print("number of sentences: {}".format(len(insts)))
        return insts

    def __len__(self):
        return len(self.insts)

    def __getitem__(self, index):
        return self.inst_ids[index]

    def collate_fn(self, batch:List[Feature]):
        word_seq_lens = [len(feature.words) for feature in batch]
        max_seq_len = max(word_seq_lens)
        max_char_seq_len = -1
        for feature in batch:
            curr_max_char_seq_len = max(feature.char_seq_lens)
            max_char_seq_len = max(curr_max_char_seq_len, max_char_seq_len)
        for i, feature in enumerate(batch):
            padding_length = max_seq_len - len(feature.words)
            words = feature.words + [0] * padding_length
            chars = []
            char_seq_lens = feature.char_seq_lens + [1] * padding_length
            for word_idx in range(feature.word_seq_len):
                pad_char_length = max_char_seq_len - feature.char_seq_lens[word_idx]
                word_chars = feature.chars[word_idx] + [0] * pad_char_length
                chars.append(word_chars)
            for _ in range(max_seq_len - feature.word_seq_len):
                chars.append([0] * max_char_seq_len)
            labels = feature.labels + [0] * padding_length if feature.labels is not None else None

            batch[i] = Feature(words=np.asarray(words),
                               chars=np.asarray(chars), char_seq_lens=np.asarray(char_seq_lens),
                               context_emb = feature.context_emb,
                               word_seq_len = feature.word_seq_len,
                               labels= np.asarray(labels) if labels is not None else None)
        results = Feature(*(default_collate(samples) if not check_all_obj_is_None(samples) else None for samples in zip(*batch) ))
        return results


# ##testing code to test the dataset loader
# train_dataset = NERDataset(file="data/conll2003_sample/train.txt",is_train=True)
# label2idx = train_dataset.label2idx
# dev_dataset = NERDataset(file="data/conll2003_sample/train.txt",is_train=False, label2idx=label2idx)
# test_dataset = NERDataset(file="data/conll2003_sample/train.txt",is_train=False, label2idx=label2idx)
#
# word2idx, _, char2idx, _ = build_word_idx(train_dataset.insts, dev_dataset.insts, test_dataset.insts)
# train_dataset.convert_instances_to_feature_tensors(word2idx=word2idx, char2idx=char2idx)
# dev_dataset.convert_instances_to_feature_tensors(word2idx=word2idx, char2idx=char2idx)
# test_dataset.convert_instances_to_feature_tensors(word2idx=word2idx, char2idx=char2idx)
#
#
# from torch.utils.data import DataLoader
# train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0, collate_fn=train_dataset.collate_fn)
# print(len(train_dataloader))
# for batch in train_dataloader:
#     print(batch.words)
#     exit(0)
#     # pass
