# 
# @author: Allan
#

from tqdm import tqdm
from typing import List, Dict
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from transformers import PreTrainedTokenizer
import collections
import numpy as np
from src.data.data_utils import convert_iobes, build_label_idx, check_all_labels_in_dict, iob2

from src.data import Instance
from operator import itemgetter

# import truecase
import re

Feature = collections.namedtuple('Feature', 'input_ids attention_mask token_type_ids orig_to_tok_index word_seq_len label_ids')
Feature.__new__.__defaults__ = (None,) * 6


# def truecase_sentence(tokens):
#     word_lst = [(w, idx) for idx, w in enumerate(tokens) if all(c.isalpha() for c in w)]
#     lst = [w for w, _ in word_lst if re.match(r'\b[A-Z\.\-]+\b', w)]
#
#     if len(lst) and len(lst) == len(word_lst):
#         parts = truecase.get_true_case(' '.join(lst)).split()
#
#         # the trucaser have its own tokenization ...
#         # skip if the number of word dosen't match
#         if len(parts) != len(word_lst): return tokens
#
#         for (w, idx), nw in zip(word_lst, parts):
#             tokens[idx] = nw


def convert_instances_to_feature_tensors(instances: List[Instance],
                                         tokenizer: PreTrainedTokenizer,
                                         label2idx: Dict[str, int], docs_all) -> List[Feature]:
    features = []
    # max_candidate_length = -1
    max_word_pieces = 510
    docs_all_word_pieces = []
    docs_all_lens = []
    for doc in docs_all:
        doc_word_pieces = []
        doc_sent_lens = []
        for sent in doc:
            sent_word_pieces = []
            for i, word in enumerate(sent):
                word_tokens = tokenizer.tokenize(" " + word)
                for sub_token in word_tokens:
                    sent_word_pieces.append(sub_token)
                    doc_word_pieces.append(sub_token)
            # doc_word_pieces.append(sent_word_pieces)
            doc_sent_lens.append(len(sent_word_pieces))
        docs_all_word_pieces.append(doc_word_pieces)
        docs_all_lens.append(doc_sent_lens)

    for idx, inst in enumerate(instances):
        doc_id, sent_id = inst.doc_idx
        words = inst.ori_words
        orig_to_tok_index = []
        tokens = []

        # truecase_sentence(words)

        for i, word in enumerate(words):
            """
            Note: by default, we use the first wordpiece token to represent the word
            If you want to do something else (e.g., use last wordpiece to represent), modify them here.
            """
            orig_to_tok_index.append(len(tokens))
            ## tokenize the word into word_piece / BPE
            ## NOTE: adding a leading space is important for BART/GPT/Roberta tokenization.
            ## Related GitHub issues:
            ##      https://github.com/huggingface/transformers/issues/1196
            ##      https://github.com/pytorch/fairseq/blob/master/fairseq/models/roberta/hub_interface.py#L38-L56
            ##      https://github.com/ThilinaRajapakse/simpletransformers/issues/458
            word_tokens = tokenizer.tokenize(" " + word)
            for sub_token in word_tokens:
                tokens.append(sub_token)
        left_length = sum(docs_all_lens[doc_id][:sent_id])
        sent_right = sum(docs_all_lens[doc_id][:sent_id+1])
        right_length = sum(docs_all_lens[doc_id][sent_id+1:])
        sentence_length = len(tokens)
        half_context_length = int((510 - sentence_length) / 2)

        if left_length < right_length:
            left_context_length = min(left_length, half_context_length)
            right_context_length = min(right_length, 510 - left_context_length - sentence_length)
        else:
            right_context_length = min(right_length, half_context_length)
            left_context_length = min(left_length, 510 - right_context_length - sentence_length)

        doc_offset = left_length - left_context_length
        target_tokens = docs_all_word_pieces[doc_id][doc_offset: sent_right + right_context_length]
        orig_to_tok_index = (np.array(orig_to_tok_index) + left_context_length).tolist()
        labels = inst.labels
        label_ids = [label2idx[label] for label in labels] if labels else [-100] * len(words)
        input_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + target_tokens + [tokenizer.sep_token])
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        assert (len(labels) == len(orig_to_tok_index))
        # print('-------------------------check------------------------')
        # # print(docs_all_word_pieces[doc_id])
        # print(target_tokens)
        # print(input_ids)
        # print(tokens)
        # print(orig_to_tok_index)
        # print(labels)
        # print(itemgetter(*orig_to_tok_index)(target_tokens))
        # input('')


        features.append(Feature(input_ids=input_ids,
                                attention_mask=input_mask,
                                orig_to_tok_index=orig_to_tok_index,
                                token_type_ids=segment_ids,
                                word_seq_len=len(orig_to_tok_index),
                                label_ids=label_ids))
    return features


class TransformersNERDatasetDoc(Dataset):

    def __init__(self, file: str,
                 tokenizer: PreTrainedTokenizer,
                 is_train: bool,
                 sents: List[List[str]] = None,
                 label2idx: Dict[str, int] = None,
                 number: int = -1):
        """
        sents: we use sentences if we want to build dataset from sentences directly instead of file
        """
        ## read all the instances. sentences and labels
        insts, docs_all = self.read_txt(file=file, number=number) if sents is None else self.read_from_sentences(sents)
        self.insts = insts
        if is_train:
            print(f"[Data Info] Using the training set to build label index")
            assert label2idx is None
            ## build label to index mapping. e.g., B-PER -> 0, I-PER -> 1
            idx2labels, label2idx = build_label_idx(insts)
            self.idx2labels = idx2labels
            self.label2idx = label2idx
        else:
            assert label2idx is not None ## for dev/test dataset we don't build label2idx
            self.label2idx = label2idx
            # check_all_labels_in_dict(insts=insts, label2idx=self.label2idx)
        self.insts_ids = convert_instances_to_feature_tensors(insts, tokenizer, label2idx, docs_all)
        self.tokenizer = tokenizer

    def read_from_sentences(self, sents: List[List[str]]):
        """
        sents = [['word_a', 'word_b'], ['word_aaa', 'word_bccc', 'word_ccc']]
        """
        insts = []
        for sent in sents:
            insts.append(Instance(words=sent, ori_words=sent))
        return insts


    def read_txt(self, file: str, number: int = -1) -> List[Instance]:
        print(f"[Data Info] Reading file: {file}, labels will be converted to IOBES encoding")
        print(f"[Data Info] Modify src/data/transformers_dataset.read_txt function if you have other requirements")
        insts = []
        count = 0
        doc_all = []
        with open(file, 'r', encoding='utf-8') as f:
            words = []
            ori_words = []
            labels = []

            docs = []
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line.startswith("-DOCSTART"):
                    if docs:
                        doc_all.append(docs)
                        docs = []
                    continue

                if line == "":
                    if words:
                        labels = convert_iobes(iob2(labels))
                        doc_idx = [len(doc_all), len(docs)] ## which document, and which sentence in the document
                        docs.append(words)
                        if count < 5:
                            count += 1
                            print('Words: ', words)
                            print('Labels: ', labels)
                        insts.append(Instance(words=words, ori_words=ori_words, labels=labels, doc_idx=doc_idx))
                        words = []
                        ori_words = []
                        labels = []


                        if len(insts) == number:
                            break
                        continue
                if not line:
                    pass
                else:
                    ls = line.split()
                    word, label = ls[0], ls[-1]
                    ori_words.append(word)
                    words.append(word)
                    labels.append(label)
        print("number of sentences: {}".format(len(insts)))
        doc_all.append(docs)
        return insts, doc_all

    def __len__(self):
        return len(self.insts_ids)

    def __getitem__(self, index):
        return self.insts_ids[index]

    def collate_fn(self, batch:List[Feature]):
        word_seq_len = [len(feature.orig_to_tok_index) for feature in batch]
        max_seq_len = max(word_seq_len)
        max_wordpiece_length = max([len(feature.input_ids) for feature in batch])
        for i, feature in enumerate(batch):
            padding_length = max_wordpiece_length - len(feature.input_ids)
            input_ids = feature.input_ids + [self.tokenizer.pad_token_id] * padding_length
            mask = feature.attention_mask + [0] * padding_length
            type_ids = feature.token_type_ids + [self.tokenizer.pad_token_type_id] * padding_length
            padding_word_len = max_seq_len - len(feature.orig_to_tok_index)
            orig_to_tok_index = feature.orig_to_tok_index + [0] * padding_word_len
            label_ids = feature.label_ids + [0] * padding_word_len

            batch[i] = Feature(input_ids=np.asarray(input_ids),
                               attention_mask=np.asarray(mask), token_type_ids=np.asarray(type_ids),
                               orig_to_tok_index=np.asarray(orig_to_tok_index),
                               word_seq_len =feature.word_seq_len,
                               label_ids=np.asarray(label_ids))
        results = Feature(*(default_collate(samples) for samples in zip(*batch)))
        return results


## testing code to test the dataset
# from transformers import *
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# dataset = TransformersNERDataset(file= "data/conll2003_sample/train.txt",tokenizer=tokenizer, is_train=True)
# from torch.utils.data import DataLoader
# train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2, collate_fn=dataset.collate_fn)
# print(len(train_dataloader))
# for batch in train_dataloader:
#     # print(batch.input_ids.size())
#     print(batch.input_ids)
#     pass
