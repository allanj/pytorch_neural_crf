import argparse
import random
import numpy as np
from src.config import Config, ContextEmb, evaluate_batch_insts, write_results
import time
from src.model import NNCRF
import torch
from typing import List
from termcolor import colored
import os
from src.config.utils import load_elmo_vec
from src.config import context_models, get_metric
import pickle
import tarfile
from tqdm import tqdm
from collections import Counter
from src.data import NERDataset
from src.data.data_utils import build_word_idx
from torch.utils.data import DataLoader
from src.config.utils import get_optimizer, lr_decay
from src.data import Instance


def set_seed(opt, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if opt.device.startswith("cuda"):
        print("using GPU...", torch.cuda.current_device())
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_arguments(parser):
    ###Training Hyperparameters
    parser.add_argument('--device', type=str, default="cpu", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'],
                        help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--dataset', type=str, default="conll2003_sample")
    parser.add_argument('--embedding_file', type=str, default="data/glove.6B.100d.txt",
                        help="we will be using random embeddings if file do not exist")
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default="sgd", help="This would be useless if you are working with transformers package")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="usually we use 0.01 for sgd but 2e-5 working with bert/roberta")
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=10, help="default batch size is 10 (works well for normal neural crf), here default 30 for bert-based crf")
    parser.add_argument('--num_epochs', type=int, default=100, help="Usually we set to 100.")
    parser.add_argument('--train_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--dev_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--test_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--max_no_incre', type=int, default=100, help="early stop when there is n epoch not increasing on dev")

    ##model hyperparameter
    parser.add_argument('--model_folder', type=str, default="english_model", help="The name to save the model files")
    parser.add_argument('--hidden_dim', type=int, default=200, help="hidden size of the LSTM, usually we set to 200 for LSTM-CRF")
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout for embedding")
    parser.add_argument('--use_char_rnn', type=int, default=1, choices=[0, 1], help="use character-level lstm, 0 or 1")
    parser.add_argument('--static_context_emb', type=str, default="none", choices=["none", "elmo"],
                        help="static contextual word embedding, our old ways to incorporate ELMo and BERT.")
    parser.add_argument('--add_iobes_constraint', type=int, choices=[0, 1], default=0,
                        help="add IOBES constraint for transition parameters to enforce valid transitions")

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def train_model(config: Config, epoch: int, train_loader: DataLoader, dev_loader: DataLoader, test_loader: DataLoader):
    ### Data Processing Info
    train_num = len(train_loader.dataset)
    print("number of instances: %d" % (train_num))


    model = NNCRF(config)
    optimizer = get_optimizer(config, model)

    dev = config.device
    model.to(config.device)

    best_dev = [-1, 0]
    best_test = [-1, 0]

    model_folder = config.model_folder
    res_folder = "results"
    if os.path.exists("model_files/" + model_folder):
        raise FileExistsError(
            f"The folder model_files/{model_folder} exists. Please either delete it or create a new one "
            f"to avoid override.")
    model_path = f"model_files/{model_folder}/lstm_crf.m"
    config_path = f"model_files/{model_folder}/config.conf"
    res_path = f"{res_folder}/{model_folder}.results"
    print("[Info] The model will be saved to: %s.tar.gz" % (model_folder))
    os.makedirs(f"model_files/{model_folder}", exist_ok= True) ## create model files. not raise error if exist
    os.makedirs(res_folder, exist_ok=True)
    no_incre_dev = 0
    print(colored(f"[Train Info] Start training, you have set to stop if performace not increase for {config.max_no_incre} epochs",'red'))
    for i in tqdm(range(1, epoch + 1), desc="Epoch"):
        epoch_loss = 0
        start_time = time.time()
        model.zero_grad()
        model.train()
        if config.optimizer.lower() == "sgd":
            optimizer = lr_decay(config, optimizer, i)
        for iter, batch in tqdm(enumerate(train_loader, 1), desc="--training batch", total=len(train_loader)):
            loss = model(words = batch.words.to(dev), word_seq_lens = batch.word_seq_len.to(dev),
                         context_emb= batch.context_emb.to(dev) if batch.context_emb is not None else None,
                         chars = batch.chars.to(dev), char_seq_lens = batch.char_seq_lens.to(dev), labels = batch.labels.to(dev))
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()
        end_time = time.time()
        print("Epoch %d: %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time), flush=True)

        model.eval()
        dev_metrics = evaluate_model(config, model, dev_loader, "dev", dev_loader.dataset.insts)
        test_metrics = evaluate_model(config, model, test_loader, "test", test_loader.dataset.insts)
        if dev_metrics[2] > best_dev[0]:
            print("saving the best model...")
            no_incre_dev = 0
            best_dev[0] = dev_metrics[2]
            best_dev[1] = i
            best_test[0] = test_metrics[2]
            best_test[1] = i
            torch.save(model.state_dict(), model_path)
            # Save the corresponding config as well.
            f = open(config_path, 'wb')
            pickle.dump(config, f)
            f.close()
            write_results(res_path, test_loader.dataset.insts)
        else:
            no_incre_dev += 1
        model.zero_grad()
        if no_incre_dev >= config.max_no_incre:
            print("early stop because there are %d epochs not increasing f1 on dev"%no_incre_dev)
            break

    print("Archiving the best Model...")
    with tarfile.open(f"model_files/{model_folder}/{model_folder}.tar.gz", "w:gz") as tar:
        tar.add(f"model_files/{model_folder}", arcname=os.path.basename(model_folder))

    print("Finished archiving the models")

    print("The best dev: %.2f" % (best_dev[0]))
    print("The corresponding test: %.2f" % (best_test[0]))
    print("Final testing.")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    evaluate_model(config, model, test_loader, "test", test_loader.dataset.insts)
    write_results(res_path, test_loader.dataset.insts)


def evaluate_model(config: Config, model: NNCRF, loader: DataLoader, name: str, insts: List[Instance], print_each_type_metric: bool = False):
    ## evaluation
    p_dict, total_predict_dict, total_entity_dict = Counter(), Counter(), Counter()
    batch_id = 0
    batch_size = loader.batch_size
    dev = config.device
    with torch.no_grad():
        for iter, batch in tqdm(enumerate(loader, 1), desc="--evaluating batch", total=len(loader)):
            one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
            batch_max_scores, batch_max_ids = model.decode(words = batch.words.to(dev), word_seq_lens = batch.word_seq_len.to(dev),
                         context_emb=batch.context_emb.to(dev) if batch.context_emb is not None else None,
                         chars = batch.chars.to(dev), char_seq_lens = batch.char_seq_lens.to(dev))
            batch_p , batch_predict, batch_total = evaluate_batch_insts(one_batch_insts, batch_max_ids, batch.labels, batch.word_seq_len, config.idx2labels)
            p_dict += batch_p
            total_predict_dict += batch_predict
            total_entity_dict += batch_total
            batch_id += 1
    if print_each_type_metric:
        for key in total_entity_dict:
            precision_key, recall_key, fscore_key = get_metric(p_dict[key], total_entity_dict[key], total_predict_dict[key])
            print(f"[{key}] Prec.: {precision_key:.2f}, Rec.: {recall_key:.2f}, F1: {fscore_key:.2f}")

    total_p = sum(list(p_dict.values()))
    total_predict = sum(list(total_predict_dict.values()))
    total_entity = sum(list(total_entity_dict.values()))
    precision, recall, fscore = get_metric(total_p, total_entity, total_predict)
    print(colored(f"[{name} set Total] Prec.: {precision:.2f}, Rec.: {recall:.2f}, F1: {fscore:.2f}", 'blue'), flush=True)


    return [precision, recall, fscore]


def main():
    parser = argparse.ArgumentParser(description="LSTM CRF implementation")
    opt = parse_arguments(parser)
    conf = Config(opt)

    set_seed(opt, conf.seed)

    print(colored(f"[Data Info] Reading dataset from: \n{conf.train_file}\n{conf.dev_file}\n{conf.test_file}", "blue"))
    train_dataset = NERDataset(conf.train_file, number=conf.train_num, is_train=True)
    conf.label2idx = train_dataset.label2idx
    conf.idx2labels = train_dataset.idx2labels
    dev_dataset = NERDataset(conf.dev_file, number=conf.dev_num,
                                         label2idx=train_dataset.label2idx, is_train=False)
    test_dataset = NERDataset(conf.test_file, number=conf.test_num,
                                          label2idx=train_dataset.label2idx, is_train=False)
    word2idx, _, char2idx, _ = build_word_idx(train_dataset.insts, dev_dataset.insts, test_dataset.insts)
    conf.word2idx = word2idx
    conf.char2idx = char2idx

    train_elmo_vecs, dev_elmo_vecs, test_elmo_vecs = None, None, None
    if conf.static_context_emb != ContextEmb.none:
        print('Loading the static ELMo vectors for all datasets.')
        train_elmo_vecs = load_elmo_vec(conf.train_file + "." + conf.static_context_emb.name + ".vec")
        conf.context_emb_size = train_elmo_vecs[0].shape[1]
        dev_elmo_vecs = load_elmo_vec(conf.dev_file + "." + conf.static_context_emb.name + ".vec")
        test_elmo_vecs = load_elmo_vec(conf.test_file + "." + conf.static_context_emb.name + ".vec")

    train_dataset.convert_instances_to_feature_tensors(word2idx=word2idx, char2idx=char2idx, elmo_vecs=train_elmo_vecs)
    dev_dataset.convert_instances_to_feature_tensors(word2idx=word2idx, char2idx=char2idx, elmo_vecs=dev_elmo_vecs)
    test_dataset.convert_instances_to_feature_tensors(word2idx=word2idx, char2idx=char2idx, elmo_vecs=test_elmo_vecs)

    num_workers = 8
    conf.label_size = len(train_dataset.label2idx)
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=num_workers,
                                 collate_fn=test_dataset.collate_fn)


    conf.build_emb_table(word2idx=word2idx)

    print("[Data Info] num words: " + str(len(word2idx)))

    train_model(conf, conf.num_epochs, train_dataloader, dev_dataloader, test_dataloader)


if __name__ == "__main__":
    main()
