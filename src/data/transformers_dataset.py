# 
# @author: Allan
#

from tqdm import tqdm
from typing import List, Dict
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from transformers import PreTrainedTokenizerFast
import collections
import numpy as np
from src.data.data_utils import convert_iobes, build_label_idx, check_all_labels_in_dict

from src.data import Instance
import logging
from transformers.tokenization_utils_base import BatchEncoding

logger = logging.getLogger(__name__)



def convert_instances_to_feature_tensors(instances: List[Instance],
                                         tokenizer: PreTrainedTokenizerFast,
                                         label2idx: Dict[str, int]) -> List[Dict]:
    features = []
    ## tokenize the word into word_piece / BPE
    ## NOTE: adding a leading space is important for BART/GPT/Roberta tokenization.
    ## Related GitHub issues:
    ##      https://github.com/huggingface/transformers/issues/1196
    ##      https://github.com/pytorch/fairseq/blob/master/fairseq/models/roberta/hub_interface.py#L38-L56
    ##      https://github.com/ThilinaRajapakse/simpletransformers/issues/458
    # assert tokenizer.add_prefix_space ## has to be true, in order to tokenize pre-tokenized input
    logger.info("[Data Info] We are not limiting the max length in tokenizer. You should be aware of that")
    for idx, inst in enumerate(instances):
        words = inst.ori_words
        orig_to_tok_index = []
        res = tokenizer.encode_plus(words, is_split_into_words=True)
        subword_idx2word_idx = res.word_ids(batch_index=0)
        prev_word_idx = -1
        for i, mapped_word_idx in enumerate(subword_idx2word_idx):
            """
            Note: by default, we use the first wordpiece/subword token to represent the word
            If you want to do something else (e.g., use last wordpiece to represent), modify them here.
            """
            if mapped_word_idx is None:## cls and sep token
                continue
            if mapped_word_idx != prev_word_idx:
                ## because we take the first subword to represent the whold word
                orig_to_tok_index.append(i)
                prev_word_idx = mapped_word_idx
        assert len(orig_to_tok_index) == len(words)
        labels = inst.labels
        label_ids = [label2idx[label] for label in labels] if labels else [-100] * len(words)
        segment_ids = [0] * len(res["input_ids"])

        features.append({"input_ids": res["input_ids"],
                         "attention_mask": res["attention_mask"],
                         "orig_to_tok_index": orig_to_tok_index,
                         "token_type_ids": segment_ids,
                         "word_seq_len": len(orig_to_tok_index),
                         "label_ids": label_ids})
    return features


class TransformersNERDataset(Dataset):

    def __init__(self, file: str,
                 tokenizer: PreTrainedTokenizerFast,
                 is_train: bool,
                 sents: List[List[str]] = None,
                 label2idx: Dict[str, int] = None,
                 number: int = -1):
        """
        sents: we use sentences if we want to build dataset from sentences directly instead of file
        """
        ## read all the instances. sentences and labels
        insts = self.read_file(file=file, number=number) if sents is None else self.read_from_sentences(sents)
        self.insts = insts
        if is_train:
            # assert label2idx is None
            if label2idx is not None:
                logger.warning(f"YOU ARE USING EXTERNAL label2idx, WHICH IS NOT BUILT FROM TRAINING SET.")
                self.label2idx = label2idx
            else:
                logger.info(f"[Data Info] Using the training set to build label index")
                ## build label to index mapping. e.g., B-PER -> 0, I-PER -> 1
                idx2labels, label2idx = build_label_idx(insts)
                self.idx2labels = idx2labels
                self.label2idx = label2idx
        else:
            assert label2idx is not None ## for dev/test dataset we don't build label2idx
            self.label2idx = label2idx
            # check_all_labels_in_dict(insts=insts, label2idx=self.label2idx)
        self.insts_ids = convert_instances_to_feature_tensors(insts, tokenizer, label2idx)
        self.tokenizer = tokenizer

    def read_from_sentences(self, sents: List[List[str]]):
        """
        sents = [['word_a', 'word_b'], ['word_aaa', 'word_bccc', 'word_ccc']]
        """
        insts = []
        for sent in sents:
            insts.append(Instance(words=sent, ori_words=sent))
        return insts


    def read_file(self, file: str, number: int = -1) -> List[Instance]:
        logger.info(f"[Data Info] Reading file: {file}, labels will be converted to IOBES encoding")
        logger.info(f"[Data Info] Modify src/data/transformers_dataset.read_txt function if you have other requirements")
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
                words.append(word)
                labels.append(label)
        logger.info(f"number of sentences: {len(insts)}")
        return insts

    def __len__(self):
        return len(self.insts_ids)

    def __getitem__(self, index):
        return self.insts_ids[index]

    def collate_fn(self, batch:List[Dict]):
        word_seq_len = [len(feature["orig_to_tok_index"]) for feature in batch]
        max_seq_len = max(word_seq_len)
        max_wordpiece_length = max([len(feature["input_ids"]) for feature in batch])
        for i, feature in enumerate(batch):
            padding_length = max_wordpiece_length - len(feature["input_ids"])
            input_ids = feature["input_ids"] + [self.tokenizer.pad_token_id] * padding_length
            mask = feature["attention_mask"] + [0] * padding_length
            type_ids = feature["token_type_ids"] + [self.tokenizer.pad_token_type_id] * padding_length
            padding_word_len = max_seq_len - len(feature["orig_to_tok_index"])
            orig_to_tok_index = feature["orig_to_tok_index"] + [0] * padding_word_len
            label_ids = feature["label_ids"] + [0] * padding_word_len

            batch[i] = {"input_ids": input_ids,
                        "attention_mask": mask,
                        "token_type_ids": type_ids,
                        "orig_to_tok_index": orig_to_tok_index,
                        "word_seq_len": feature["word_seq_len"],
                        "label_ids": label_ids}
        encoded_inputs = {key: [example[key] for example in batch] for key in batch[0].keys()}
        results = BatchEncoding(encoded_inputs, tensor_type='pt')
        return results


## testing code to test the dataset
if __name__ == '__main__':
    from transformers import RobertaTokenizerFast
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)
    dataset = TransformersNERDataset(file="data/conll2003_sample/train.txt",tokenizer=tokenizer, is_train=True)
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2, collate_fn=dataset.collate_fn)
    print(len(train_dataloader))
    for batch in train_dataloader:
        # print(batch.input_ids.size())
        print(batch.input_ids)
        pass
