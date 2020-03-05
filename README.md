## LSTM-CRF Model for Named Entity Recognition (or Sequence Labeling)

This repository implements an LSTM-CRF model for named entity recognition. The model is same as the one by [Lample et al., (2016)](http://www.anthology.aclweb.org/N/N16/N16-1030.pdf) except we do not have the last `tanh` layer after the BiLSTM.
We achieve the SOTA performance on both CoNLL-2003 and OntoNotes 5.0 English datasets (check our [benchmark](/docs/benchmark.md)). 

### Requirements
* Python >= 3.6 and PyTorch >= 0.4.1
* AllenNLP package (if you use ELMo)

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



### Using ELMo (and BERT) as contextualized word embeddings
There are two ways to import the ELMo and BERT representations. We can either __preprocess the input files into vectors and load them in the program__ or __use the ELMo/BERT model to _forward_ the input tokens everytime__. The latter approach allows us to fine tune the parameters in ELMo and BERT. But the memory consumption is pretty high. For the purpose of most practical use case, I simply implemented the first method.
1. Run the script with `python -m preprocess.get_elmo_vec YourData`. As a result, you get the vector files for your datasets.
2. Run the main file with command: `python trainer.py --context_emb elmo`. You are good to go.

For using BERT, it would be a similar manner. Let me know if you want further functionality. Note that, we concatenate ELMo and word embeddings (i.e., Glove) in our model (check [here](https://github.com/allanj/pytorch_lstmcrf/blob/master/model/lstmcrf.py#L82)). You may not need concatenation for BERT.

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

    





### Ongoing Plan

- [x] Support for ELMo as features
- [x] Interactive model where we can just import model and decode a setence
- [x] Make the code more modularized (separate the encoder and inference layers) and readable (by adding more comments)
- [x] Put the benchmark performance documentation to another markdown file
- [ ] Integrate ELMo/BERT as a module instead of just features.
- [ ] Clean up the code to better organization (e.g., `import` stuff)

### Contributors
A huge thanks to [@yuchenlin](https://github.com/yuchenlin) for his contribution in this repo.
