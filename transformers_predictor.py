
import pickle

from src.model import TransformersCRF
import torch
from termcolor import colored
from src.config import context_models
from src.data import TransformersNERDataset
from typing import List, Union, Tuple
import tarfile
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


class TransformersNERPredictor:

    def __init__(self, model_archived_file:str,
                 cuda_device: str = "cpu"):
        """
        model_archived_file: ends with "tar.gz"
        OR
        directly use the model folder patth
        """
        device = torch.device(cuda_device)
        if model_archived_file.endswith("tar.gz"):
            tar = tarfile.open(model_archived_file)
            self.conf = pickle.load(tar.extractfile(tar.getnames()[1])) ## config file
            self.model = TransformersCRF(self.conf)
            self.model.load_state_dict(torch.load(tar.extractfile(tar.getnames()[2]), map_location=device)) ## model file
        else:
            folder_name = model_archived_file
            assert os.path.isdir(folder_name)
            f = open(folder_name + "/config.conf", 'rb')
            self.conf = pickle.load(f)
            f.close()
            self.model = TransformersCRF(self.conf)
            self.model.load_state_dict(torch.load(f"{folder_name}/lstm_crf.m", map_location=device))
        self.conf.device = device
        self.model.to(device)
        self.model.eval()

        print(colored(f"[Data Info] Tokenizing the instances using '{self.conf.embedder_type}' tokenizer", "blue"))
        self.tokenizer = context_models[self.conf.embedder_type]["tokenizer"].from_pretrained(self.conf.embedder_type)

    def predict(self, sents: List[List[str]], batch_size = -1):
        batch_size = len(sents) if batch_size == -1 else batch_size

        dataset = TransformersNERDataset(file=None, sents=sents, tokenizer=self.tokenizer, label2idx=self.conf.label2idx, is_train=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

        all_predictions = []
        for batch_id, batch in tqdm(enumerate(loader, 0), desc="--evaluating batch", total=len(loader)):
            one_batch_insts = dataset.insts[batch_id * batch_size:(batch_id + 1) * batch_size]
            batch_max_scores, batch_max_ids = self.model.decode(words= batch.input_ids.to(self.conf.device),
                    word_seq_lens = batch.word_seq_len.to(self.conf.device),
                    orig_to_tok_index = batch.orig_to_tok_index.to(self.conf.device),
                    input_mask = batch.attention_mask.to(self.conf.device))

            for idx in range(len(batch_max_ids)):
                length = batch.word_seq_len[idx]
                prediction = batch_max_ids[idx][:length].tolist()
                prediction = prediction[::-1]
                prediction = [self.conf.idx2labels[l] for l in prediction]
                one_batch_insts[idx].prediction = prediction
                all_predictions.append(prediction)
        return all_predictions


if __name__ == '__main__':
    sents = [
        ['I', 'am', 'traveling', 'to', 'Singapore', 'to', 'visit', 'the', 'Merlion', 'Park', '.'],
        ['John', 'cannot', 'come', 'with', 'us', '.']
    ]
    model_path = "model_files/english_model"
    device = "cpu" # cpu, cuda:0, cuda:1
    ## or model_path = "model_files/english_model.tar.gz"
    predictor = TransformersNERPredictor(model_path, cuda_device=device)
    prediction = predictor.predict(sents)
    print(prediction)
