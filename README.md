## LSTM/BERT-CRF Model for Named Entity Recognition (or Sequence Labeling)

This repository implements an LSTM-CRF model for named entity recognition. The model is same as the one by [Lample et al., (2016)](http://www.anthology.aclweb.org/N/N16/N16-1030.pdf) except we do not have the last `tanh` layer after the BiLSTM.
We achieve the SOTA performance on both CoNLL-2003 and OntoNotes 5.0 English datasets (check our [benchmark](/docs/benchmark.md) with Glove and ELMo, other 
and [benchmark results](/docs/transformers_benchmark.md) with fine-tuning BERT). 

**Announcements**
* We implemented distributed training for faster training
* We implemented a [**Faster CRF**](/docs/fast_crf.md) module which allows **O(log N) inference and back-tracking**! 
* [Benchmark results](/docs/transformers_benchmark.md) by fine-tuning BERT/Roberta**


| Model| Dataset | Precision | Recall | F1 |
|-------| ------- | :---------: | :------: | :--: |
|BERT-base-cased + CRF (this repo)| CONLL-2003 | 91.69 | 92.05 | 91.87 |
|Roberta-base  + CRF (this repo)| CoNLL-2003 | **91.88**  | **93.01** |**92.44**|
|BERT-base-cased  + CRF (this repo)| OntoNotes 5 |89.57  | 89.45 | 89.51 |
|Roberta-base  + CRF (this repo)| OntoNotes 5 | **90.12**  | **91.25** |**90.68**|

More [details](/docs/transformers_benchmark.md)

### Requirements
* Python >= 3.6 and PyTorch >= 1.6.0 (tested)
* pip install transformers
* pip install accelerate (optional for distributed training)
* pip install seqeval (optional, only used in evaluation while in distributed training)

In the documentation below, we present two ways for users to run the code:
1. Run the model via (Fine-tuning) BERT/Roberta/etc in Transformers package.
2. Run the model with simply word embeddings (and static ELMo/BERT representations loaded from external vectors).

Our default argument setup refers to the first one `1`.

### Usage with Fine-Tuning BERT/Roberta (,etc) models in HuggingFace
1. Simply replace the `embedder_type` argument with the model in HuggingFace. For example, if we are using `roberta-large`, we just need to 
change the embedder type as `roberta-large`. 
    ```bash
    python transformers_trainer.py --device=cuda:0 --dataset=YourData --model_folder=saved_models --embedder_type=roberta-base
    ```

2. **Distributed Training** (If necessary)
   1. We use huggingface `accelerate` package to enable distributed training. After you set the proper configuration of your distributed environment,
      by `accelerate config`, you can easily run the following command for distributed training
    ```bash
    accelerate launch transformers_trainer_ddp.py --batch_size=30 {YOUR_OTHER_ARGUMENTS}
    ```
   Note that, this `batch size` is actually __batch_size per GPU device__.

3. **(Optional) Using other models in HuggingFace.**
    1.  Run the main file with modified argument `embedder_type`:
        ```bash
        python trainer.py --embedder_type=bert-large-cased
        ```
        The default value for `embedder_type` is `roberta-base`.
        Changing the name to something like `bert-base-cased` or `roberta-large`, we directly load the model from huggingface.
        **Note**: if you use other models, remember to replace the [tokenization mechanism]() in `config/utils.py`.
       
        Our default tokenizer is assumed to be `fast_tokenizer`. If your tokenizer does not support `fast` mode, try set `use_fast=False`:
        ```python3
        tokenizer = AutoTokenizer.from_pretrained(conf.embedder_type, add_prefix_space=True, use_fast=False)
        ```
    2. Finally, if you would like to know more about the details, read more details below:
        * [Tokenization](/docs/bert_tokenization.md): For BERT, we use the first wordpice to represent a complete word. Check `config/transformers_util.py`
        * [Embedder](/docs/bert_embedder.md): We show how to embed the input tokens to make word representation. Check `model/embedder/transformers_embedder.py`
    3. Using BERT/Roberta as contextualized word embeddings (Static, Feature-based Approach)
       Simply go to `model/transformers_embedder.py` and uncomment the following:
       ```python
       self.model.requires_grad = False
       ```


### Other Usages
Using Word embedding or external contextualized embedding (ELMo/BERT) can be found in [here](/docs/other_usage.md).


### Training with your own data. 
1. Create a folder `YourData` under the data directory. 
2. Put the `train.txt`, `dev.txt` and `test.txt` files (make sure the format is compatible, i.e. the first column is words and the last column are tags) under this directory.  If you have a different format, simply modify the reader in `config/reader.py`. 
3. Change the `dataset` argument to `YourData` when you run `trainer.py`. 






### Further Details and Extensions

1. [Benchmark Performance](/docs/benchmark.md)
2. [Benchmark on BERT/Roberta](/docs/transformers_benchmark.md)






### Ongoing Plan

- [x] Support for ELMo/BERT as features
- [x] Interactive model where we can just import model and decode a setence
- [x] Make the code more modularized (separate the encoder and inference layers) and readable (by adding more comments)
- [x] Put the benchmark performance documentation to another markdown file
- [x] Integrate BERT as a module instead of just features.
- [x] Clean up the code to better organization (e.g., `import` stuff)
- [x] Benchmark experiments for Transformers' based models.
- [x] Support FP-16 training/inference
- [x] Support distributed training using accelerate
- [ ] Releases some pre-trained NER models. 
- [ ] Semi-CRF model support 


