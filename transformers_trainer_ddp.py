import argparse
from src.config import Config, from_label_id_tensor_to_label_sequence
import time
from src.model import TransformersCRF
import torch
import os
from src.config.utils import write_results
from src.config.transformers_util import get_huggingface_optimizer_and_scheduler
import pickle
import tarfile
from tqdm import tqdm
from collections import Counter
from src.data import TransformersNERDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from accelerate.utils import set_seed
import logging
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from src.data.data_utils import PAD
import datasets
from datasets.metric import Metric

"""
Same as transformers_trainer.py but with distributed training.
"""

from accelerate import DistributedDataParallelKwargs

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

tqdm = partial(
    tqdm,
    bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    disable=not accelerator.is_local_main_process,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = get_logger(__name__)
logger.setLevel(logging.INFO)


def parse_arguments(parser):
    ###Training Hyperparameters
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--dataset", type=str, default="conll2003_sample")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        help="This would be useless if you are working with transformers package",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="usually we use 0.01 for sgd but 2e-5 working with bert/roberta",
    )
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--l2", type=float, default=1e-8)
    parser.add_argument("--lr_decay", type=float, default=0)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=30,
        help="batch_size per device. For distributed training, this is the batch_size per gpu",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Usually we set to 100."
    )
    parser.add_argument(
        "--train_num", type=int, default=-1, help="-1 means all the data"
    )
    parser.add_argument("--dev_num", type=int, default=-1, help="-1 means all the data")
    parser.add_argument(
        "--test_num", type=int, default=-1, help="-1 means all the data"
    )
    parser.add_argument(
        "--max_no_incre",
        type=int,
        default=80,
        help="early stop when there is n epoch not increasing on dev",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="The maximum gradient norm, if <=0, means no clipping, usually we don't use clipping for normal neural ncrf",
    )
    parser.add_argument(
        "--fp16",
        type=int,
        choices=[0, 1],
        default=1,
        help="use 16-bit floating point precision instead of 32-bit",
    )

    ##model hyperparameter
    parser.add_argument(
        "--model_folder",
        type=str,
        default="english_model",
        help="The name to save the model files",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=0,
        help="hidden size of the LSTM, usually we set to 200 for LSTM-CRF",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout for embedding"
    )

    parser.add_argument(
        "--embedder_type",
        type=str,
        default="roberta-base",
        help="you can use 'bert-base-uncased' and so on",
    )
    parser.add_argument(
        "--add_iobes_constraint",
        type=int,
        default=0,
        choices=[0, 1],
        help="add IOBES constraint for transition parameters to enforce valid transitions",
    )

    parser.add_argument(
        "--print_detail_f1",
        type=int,
        default=0,
        choices=[0, 1],
        help="Open and close printing f1 scores for each tag after each evaluation epoch",
    )
    parser.add_argument(
        "--earlystop_atr",
        type=str,
        default="micro",
        choices=["micro", "macro"],
        help="Choose between macro f1 score and micro f1 score for early stopping evaluation",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="training model or test mode",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/conll2003_sample/test.txt",
        help="test file for test mode, only applicable in test mode",
    )

    args = parser.parse_args()
    for k in args.__dict__:
        logger.info(k + ": " + str(args.__dict__[k]))
    return args


def train_model(
    config: Config,
    epoch: int,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    test_loader: DataLoader,
):
    ### Data Processing Info
    train_num = len(train_loader)
    logger.info(f"[Data Info] number of training instances: {train_num}")

    logger.info(
        f"[Model Info]: Working with transformers package from huggingface with {config.embedder_type}"
    )
    logger.info(
        f"[Optimizer Info]: You should be aware that you are using the optimizer from huggingface."
    )
    logger.info(
        f"[Optimizer Info]: Change the optimier in transformers_util.py if you want to make some modifications."
    )
    model = TransformersCRF(config)
    optimizer, scheduler = get_huggingface_optimizer_and_scheduler(
        model=model,
        learning_rate=config.learning_rate,
        num_training_steps=len(train_loader) * epoch,
        weight_decay=0.0,
        eps=1e-8,
        warmup_step=0,
    )
    logger.info(f"[Optimizer Info] Modify the optimizer info as you need.")
    logger.info(optimizer)

    model, optimizer, train_loader, dev_loader, test_loader, scheduler = (
        accelerator.prepare(
            model, optimizer, train_loader, dev_loader, test_loader, scheduler
        )
    )

    metric = datasets.load_metric("seqeval")
    best_dev = [-1, 0]
    best_test = [-1, 0]

    model_folder = config.model_folder
    res_folder = "results"
    if os.path.exists("model_files/" + model_folder):
        raise FileExistsError(
            f"The folder model_files/{model_folder} exists. Please either delete it or create a new one "
            f"to avoid override."
        )
    model_path = f"model_files/{model_folder}/lstm_crf.m"
    config_path = f"model_files/{model_folder}/config.conf"
    res_path = f"{res_folder}/{model_folder}.results"
    logger.info("[Info] The model will be saved to: %s.tar.gz" % (model_folder))
    os.makedirs(
        f"model_files/{model_folder}", exist_ok=True
    )  ## create model files. not raise error if exist
    os.makedirs(res_folder, exist_ok=True)
    no_incre_dev = 0
    logger.info(
        f"[Train Info] Start training, you have set to stop if performace not increase for {config.max_no_incre} epochs"
    )
    for i in tqdm(range(1, epoch + 1), desc="Epoch"):
        epoch_loss = 0
        start_time = time.time()
        model.zero_grad()
        model.train()
        for iter, batch in tqdm(
            enumerate(train_loader, 1), desc="--training batch", total=len(train_loader)
        ):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=bool(config.fp16)):
                loss = model(
                    subword_input_ids=batch.input_ids,
                    word_seq_lens=batch.word_seq_len,
                    orig_to_tok_index=batch.orig_to_tok_index,
                    attention_mask=batch.attention_mask,
                    labels=batch.label_ids,
                ).sum()
            epoch_loss += loss.item()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()
        end_time = time.time()
        logger.info(
            f"Epoch {i}: {epoch_loss:.5f}, Time is {(end_time - start_time):.2f}s"
        )

        model.eval()
        dev_metrics = evaluate_model(config, model, dev_loader, "dev", metric)
        test_metrics = evaluate_model(config, model, test_loader, "test", metric)
        if dev_metrics[2] > best_dev[0]:
            logger.info(
                f"saving the best model with best dev f1 score {dev_metrics[2]}"
            )
            no_incre_dev = 0
            best_dev[0] = dev_metrics[2]
            best_dev[1] = i
            best_test[0] = test_metrics[2]
            best_test[1] = i
            if accelerator.is_main_process:
                module = model.module if hasattr(model, "module") else model
                torch.save(module.state_dict(), model_path)
                # Save the corresponding config as well.
                f = open(config_path, "wb")
                pickle.dump(config, f)
                f.close()
                write_results(res_path, test_loader.dataset.insts)
        else:
            no_incre_dev += 1
        model.zero_grad()
        if no_incre_dev >= config.max_no_incre:
            logger.info(
                "early stop because there are %d epochs not increasing f1 on dev"
                % no_incre_dev
            )
            break

    logger.info("Archiving the best Model...")
    with tarfile.open(f"model_files/{model_folder}.tar.gz", "w:gz") as tar:
        tar.add(f"model_files/{model_folder}", arcname=os.path.basename(model_folder))

    logger.info("Finished archiving the models")

    logger.info("The best dev: %.2f" % (best_dev[0]))
    logger.info("The corresponding test: %.2f" % (best_test[0]))
    logger.info("Final testing.")
    module = model.module if hasattr(model, "module") else model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    evaluate_model(config, model, test_loader, "test", metric)
    write_results(res_path, test_loader.dataset.insts)


def evaluate_model(
    config: Config,
    model: TransformersCRF,
    data_loader: DataLoader,
    name: str,
    metric: Metric,
    print_each_type_metric: bool = False,
):
    ## evaluation
    all_predictions = []
    all_golds = []
    insts = data_loader.dataset.insts
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=bool(config.fp16)):
        for batch_id, batch in tqdm(
            enumerate(data_loader, 0), desc="--evaluating batch", total=len(data_loader)
        ):
            batch_max_scores, batch_max_ids = model(
                subword_input_ids=batch.input_ids,
                word_seq_lens=batch.word_seq_len,
                orig_to_tok_index=batch.orig_to_tok_index,
                attention_mask=batch.attention_mask,
                is_train=False,
            )
            batch_max_ids = accelerator.pad_across_processes(
                batch_max_ids, dim=1, pad_index=config.label2idx[PAD]
            )
            batch_max_ids = accelerator.gather_for_metrics(batch_max_ids)

            batch_label_ids = accelerator.pad_across_processes(
                batch.label_ids, dim=1, pad_index=config.label2idx[PAD]
            )
            batch_label_ids = accelerator.gather_for_metrics(batch_label_ids)

            word_seq_lens = accelerator.gather_for_metrics(batch.word_seq_len)
            predict_sequences = from_label_id_tensor_to_label_sequence(
                batch_ids=batch_max_ids,
                word_seq_lens=word_seq_lens,
                need_to_reverse=True,
                idx2label=config.idx2labels,
            )
            all_predictions.extend(predict_sequences)
            gold_sequences = from_label_id_tensor_to_label_sequence(
                batch_ids=batch_label_ids,
                word_seq_lens=word_seq_lens,
                need_to_reverse=False,
                idx2label=config.idx2labels,
            )
            all_golds.extend(gold_sequences)

    results = metric.compute(
        predictions=all_predictions, references=all_golds, scheme="IOBES"
    )
    for inst, pred_seq in zip(insts, all_predictions):
        inst.prediction = pred_seq
    f1Scores = []
    if (
        print_each_type_metric
        or config.print_detail_f1
        or (config.earlystop_atr == "macro")
    ):
        for key in results:
            precision_key, recall_key, fscore_key = (
                results[key]["precision"] * 100,
                results[key]["recall"] * 100,
                results[key]["f1"] * 100,
            )
            logger.info(
                f"[{key}] Prec.: {precision_key:.2f}, Rec.: {recall_key:.2f}, F1: {fscore_key:.2f}"
            )
            f1Scores.append(fscore_key)
        if len(f1Scores) > 0:
            logger.info(
                f"[{name} set Total] Macro F1: {sum(f1Scores) / len(f1Scores):.2f}"
            )

    precision, recall, fscore = (
        results["overall_precision"] * 100,
        results["overall_recall"] * 100,
        results["overall_f1"] * 100,
    )
    logger.info(
        f"[{name} set Total] Prec.: {precision:.2f}, Rec.: {recall:.2f}, Micro F1: {fscore:.2f}"
    )

    if config.earlystop_atr == "macro" and len(f1Scores) > 0:
        fscore = sum(f1Scores) / len(f1Scores)

    return [precision, recall, fscore]


def main():
    parser = argparse.ArgumentParser(description="Transformer CRF implementation")
    opt = parse_arguments(parser)
    set_seed(opt.seed)
    metric = datasets.load_metric("seqeval")
    if opt.mode == "train":
        conf = Config(opt)
        logger.info(
            f"[Data Info] Tokenizing the instances using '{conf.embedder_type}' tokenizer"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            conf.embedder_type, add_prefix_space=True, use_fast=True
        )
        logger.info(
            f"[Data Info] Reading dataset from: \n{conf.train_file}\n{conf.dev_file}\n{conf.test_file}"
        )
        train_dataset = TransformersNERDataset(
            conf.train_file, tokenizer, number=conf.train_num, is_train=True
        )
        conf.label2idx = train_dataset.label2idx
        conf.idx2labels = train_dataset.idx2labels

        dev_dataset = TransformersNERDataset(
            conf.dev_file,
            tokenizer,
            number=conf.dev_num,
            label2idx=train_dataset.label2idx,
            is_train=False,
        )
        test_dataset = TransformersNERDataset(
            conf.test_file,
            tokenizer,
            number=conf.test_num,
            label2idx=train_dataset.label2idx,
            is_train=False,
        )
        num_workers = 8
        conf.label_size = len(train_dataset.label2idx)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=conf.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_dataset.collate_fn,
        )
        dev_dataloader = DataLoader(
            dev_dataset,
            batch_size=conf.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=dev_dataset.collate_fn,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=conf.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=test_dataset.collate_fn,
        )

        train_model(
            conf, conf.num_epochs, train_dataloader, dev_dataloader, test_dataloader
        )
    else:
        folder_name = f"model_files/{opt.model_folder}"
        assert os.path.isdir(folder_name)
        f = open(folder_name + "/config.conf", "rb")
        saved_config = pickle.load(
            f
        )  # we use `label2idx` from old config, but test file, test number
        f.close()
        logger.info(
            f"[Data Info] Tokenizing the instances using '{saved_config.embedder_type}' tokenizer"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            saved_config.embedder_type, add_prefix_space=True, use_fast=True
        )
        test_dataset = TransformersNERDataset(
            opt.test_file,
            tokenizer,
            number=opt.test_num,
            label2idx=saved_config.label2idx,
            is_train=False,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=test_dataset.collate_fn,
        )
        model = TransformersCRF(saved_config)
        model.load_state_dict(
            torch.load(f"{folder_name}/lstm_crf.m", map_location=torch.device("cpu"))
        )
        model, test_loader = accelerator.prepare(model, test_dataloader)
        model.eval()
        evaluate_model(
            config=saved_config,
            model=model,
            data_loader=test_dataloader,
            name="test mode",
            metric=metric,
            print_each_type_metric=False,
        )


if __name__ == "__main__":
    main()
