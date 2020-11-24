## LSTM-CRF Model for Named Entity Recognition (or Sequence Labeling)

This repository implements an LSTM-CRF model for named entity recognition. The model is same as the one by [Lample et al., (2016)](http://www.anthology.aclweb.org/N/N16/N16-1030.pdf) except we do not have the last `tanh` layer after the BiLSTM.
We achieve the SOTA performance on both CoNLL-2003 and OntoNotes 5.0 English datasets (check our [benchmark](/docs/benchmark.md) with Glove and ELMo, other 
and [benchmark results](/docs/transformers_benchmark.md) with fine-tuning BERT). 

**Announcement: [Benchmark results](/docs/transformers_benchmark.md) by fine-tuning BERT/Roberta**

| Model| Dataset | Precision | Recall | F1 |
|-------| ------- | :---------: | :------: | :--: |
|BERT-base-cased + CRF (this repo)| CONLL-2003 | 91.69 | 92.05 | 91.87 |
|Roberta-base  + CRF (this repo)| CoNLL-2003 | **91.88**  | **93.01** |**92.44**|
|BERT-base-cased  + CRF (this repo)| OntoNotes 5 |89.57  | 89.45 | 89.51 |
|Roberta-base  + CRF (this repo)| OntoNotes 5 | **90.12**  | **91.25** |**90.68**|

More [details](/docs/transformers_benchmark.md)

**Update**: Our latest breaking change: using data loader to read all data and convert the data into tensor. 
We latest [release](https://github.com/allanj/pytorch_lstmcrf/tree/v0.2.0) also use HuggingFace's transformers but we didn't adopt to use the PyTorch 
`Dataset` and `DataLoader` yet. This version uses both and we are also testing the correctness of the code before publishing a new release.

### Requirements
* Python >= 3.6 and PyTorch >= 1.6.0 (tested)
* Transformers package from Huggingface (Required by using Transformers)

If you use `conda`:

```bash
git clone https://github.com/allanj/pytorch_lstmcrf.git

# python > 3.6
conda create -n pt_lstmcrf python=3.6
conda activate pt_lstmcrf
# kindly check https://pytorch.org for the suitable version of your machines
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch -n pt_lstmcrf
pip install tqdm
pip install termcolor
pip install overrides
pip install allennlp ## required when we need to get the ELMo vectors
pip install transformers
```

In the documentation below, we present two ways for users to run the code:
1. Run the model via (Fine-tuning) BERT/Roberta/etc in Transformers package.
2. Run the model with simply word embeddings (and static ELMo/BERT representations loaded from external vectors).

Our default argument setup refers to the first one `1`.

### Usage with Fine-Tuning BERT/Roberta (,etc) models in HuggingFace
1. Simply replace the `embedder_type` argument with the model in HuggingFace. For example, if we are using `bert-base-cased`, we just need to 
change the embedder type as `bert-base-cased`. 
    ```bash
    python transformers_trainer.py --device=cuda:0 --dataset=YourData --model_folder=saved_models --embedder_type=bert-base-cased
    ```
2. **(Optional) Using other models in HuggingFace.**
    1. Check if your prefered language model in `config/transformers_util.py`. If not, add to the utils. For example, if you would like to use `BERT-Large`. Add the following line to the dictionary.
        ```python
           'bert-large-cased' : {  "model": BertModel,  "tokenizer" : BertTokenizer }
        ```
        This name `bert-large-cased` has to follow the naming rule by HuggingFace.
    2. Run the main file with modified argument `embedder_type`:
        ```bash
           python trainer.py --embedder_type=bert-large-cased
        ```
        The default value for `embedder_type` is `normal`, which refers to the classic LSTM-CRF and we can use `static_context_emb` in previous section.
        Changing the name to something like `bert-base-cased` or `roberta-base`, we directly load the model from huggingface.
        **Note**: if you use other models, remember to replace the [tokenization mechanism]() in `config/utils.py`.
    3.  Finally, if you would like to know more about the details, read more details below:
        * [Tokenization](/docs/bert_tokenization.md): For BERT, we use the first wordpice to represent a complete word. Check `config/transformers_util.py`
        * [Embedder](/docs/bert_embedder.md): We show how to embed the input tokens to make word representation. Check `model/embedder/transformers_embedder.py`
    4. Using BERT/Roberta as contextualized word embeddings (Static, Feature-based Approach)
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
- [ ] Releases some pre-trained NER models. 
- [ ] Support FP-16 training/inference
- [ ] Semi-CRF model support 

### Contributors
A huge thanks to [@yuchenlin](https://github.com/yuchenlin) for his contribution in this repo.
