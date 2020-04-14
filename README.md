## LSTM-CRF Model for Named Entity Recognition (or Sequence Labeling)

This repository implements an LSTM-CRF model for named entity recognition. The model is same as the one by [Lample et al., (2016)](http://www.anthology.aclweb.org/N/N16/N16-1030.pdf) except we do not have the last `tanh` layer after the BiLSTM.
We achieve the SOTA performance on both CoNLL-2003 and OntoNotes 5.0 English datasets (check our [benchmark](/docs/benchmark.md)). 

### Requirements
* Python >= 3.6 and PyTorch >= 0.4.1
* AllenNLP package (if you use ELMo)
* Transformers package from Huggingface (Required by using Transformers)

If you use `conda`:

```bash
git clone https://github.com/allanj/pytorch_lstmcrf.git

conda create -n pt_lstmcrf python=3.7
conda activate pt_lstmcrf
# check https://pytorch.org for the suitable version of your machines
conda install pytorch=1.3.0 torchvision cudatoolkit=10.0 -c pytorch -n pt_lstmcrf
pip install tqdm
pip install termcolor
pip install overrides
pip install allennlp
pip install transformers
```

### Usage
1. Put the Glove embedding file (`glove.6B.100d.txt`) under `data` directory (You can also use ELMo/BERT/Flair, Check below.) Note that if your embedding file does not exist, we just randomly initalize the embeddings.
2. Simply run the following command and you can obtain results comparable to the benchmark above.
    ```bash
    python trainer.py
    ```
    If you want to use your 1st GPU device `cuda:0` and train models for your own dataset with elmo embedding:
    ```
    python trainer.py --device cuda:0 --dataset YourData --context_emb elmo --model_folder saved_models
    ```

##### Training with your own data. 
1. Create a folder `YourData` under the data directory. 
2. Put the `train.txt`, `dev.txt` and `test.txt` files (make sure the format is compatible, i.e. the first column is words and the last column are tags) under this directory.  If you have a different format, simply modify the reader in `config/reader.py`. 
3. Change the `dataset` argument to `YourData` when you run `trainer.py`. 


### Using ELMo (and BERT) as contextualized word embeddings (Static, Feature-based Approach)
There are two ways to import the ELMo and BERT representations. We can either __preprocess the input files into vectors and load them in the program__ or __use the ELMo/BERT model to _forward_ the input tokens everytime__. The latter approach allows us to fine tune the parameters in ELMo and BERT. But the memory consumption is pretty high. For the purpose of most practical use case, I simply implemented the first method.
1. Run the script with `python -m preprocess.get_elmo_vec YourData`. As a result, you get the vector files for your datasets.
2. Run the main file with command: `python trainer.py --static_context_emb elmo`. You are good to go.

For using BERT, it would be a similar manner. We recommend you use the [BERT-As-Service](https://github.com/hanxiao/bert-as-service) package for this purpose. (Note that you have to deal with the conversion between the wordpiece and word representations.)
Note that, we concatenate ELMo and word embeddings (i.e., Glove) in our model (check [here](https://github.com/allanj/pytorch_lstmcrf/blob/master/model/lstmcrf.py#L82)). You may not need concatenation for BERT.

### Fine-tune with BERT Embedding (Fine-tuning Approach)
In this scenario, instead of using the `NNCRF` class in [`neuralcrf.py`](/model/neuralcrf.py), we will be using the `BertNNCRF` class in [`bert_neuralcrf.py`](/model/bert_neuralcrf.py).
1. Check if your prefered language model in `config/transformers_util.py`. If not, add to the utils. For example, if you would like to use `BERT-Large`. Add the following line to the dictionary.
    ```python
       'bert-large-cased' : {  "model": BertModel,  "tokenizer" : BertTokenizer }
    ```
    This name `bert-large-cased` has to follow the naming rule by HuggingFace.
2. Run the main file with this command:
    ```bash
       python trainer.py --embedder_type bert-large-cased
    ```
    The default value for `embedder_type` is `normal`, which refers to the classic LSTM-CRF and we can use `static_context_emb` in previous section.
    Changing the name to something like `bert-base-cased` or `bert-base-uncased`, we directly load the model from huggingface.
    **Note**: if you use other models, remember to replace the [tokenization mechanism]() in `config/utils.py`.
3. (**Optional**) Finally, if you would like to know more about the details, read more details below:
    * [Tokenization](/docs/bert_tokenization.md): For BERT, we use the first wordpice to represent a complete word
    * [Embedder](/docs/bert_embedder.md): We show how to embed the input tokens to make word representation.




### Running with our pretrained English (with ELMo) Model
We trained an English LSTM-CRF (+ELMo) model on the CoNLL-2003 dataset.
You can directly predict a sentence with the following piece of code (*Note*: we do not do tokenization.).

You can download the English model through this [link](https://drive.google.com/file/d/1N1DiS9Xhjprn4cfNvIgs9GWSHC47n25C/view?usp=sharing).
```python
from ner_predictor import NERPredictor
sentence = "This is an English model ."
# Or you can make a list of sentence:
# sentence = ["This is an English model", "This is the second sentence"]
model_path = "english_model.tar.gz"
predictor = NERPredictor(model_path, cuda_device="cpu") ## you can use "cuda:0", "cuda:1" for gpu
prediction = predictor.predict(sentence)
print(prediction)
```









### Further Details and Extensions

1. [Benchmark Performance](/docs/benchmark.md)
2. Our common practice for NER is actually using ELMo is easier for tunning and obtaining quite good performance compared to BERT. But we did not try other language models.






### Ongoing Plan

- [x] Support for ELMo as features
- [x] Interactive model where we can just import model and decode a setence
- [x] Make the code more modularized (separate the encoder and inference layers) and readable (by adding more comments)
- [x] Put the benchmark performance documentation to another markdown file
- [x] Integrate BERT as a module instead of just features.
- [ ] Integrate ELMo as a module for fine-tuning.
- [x] Clean up the code to better organization (e.g., `import` stuff)

### Contributors
A huge thanks to [@yuchenlin](https://github.com/yuchenlin) for his contribution in this repo.
