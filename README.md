## LSTM-CRF Model for Named Entity Recognition

This repository implements an LSTM-CRF model for named entity recognition. The model is same as the one by [Lample et al., (2016)](http://www.anthology.aclweb.org/N/N16/N16-1030.pdf) except we do not have the last `tanh` layer after the BiLSTM.

### Requirements
* Python 3.6
* Tested on PyTorch >=0.4.1


### Usage
1. Put the Glove embedding file (`glove.6B.100d.txt`) under `data` directory (You can also use ELMo/BERT/Flair, Check below.) Note that if your embedding file does not exist, we just randomly initalize the embeddings.
2. Simply run the following command and you can obtain results comparable to the benchmark above.
    ```bash
    python3.6 trainer.py
    ```

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
predictor = NERPredictor(model_path)
prediction = predictor.predict(sentence)
print(prediction)
```


### Further Details and Extensions

1. [Benchmark Performance](/docs/benchmark.md)
2. [Using ELMo (and BERT)](/docs/context_emb.md)
3. [Running with your own data](/docs/customized.md)

    





### Ongoing Plan

- [x] Support for ELMo as features
- [x] Interactive model where we can just import model and decode a setence
- [ ] Make the code more modularized (separate the encoder and inference layers) and readable (by adding more comments)
- [ ] Put the benchmark performance documentation to another markdown file
- [ ] Integrate ELMo/BERT as a module instead of just features.
- [ ] Clean up the code to better organization (e.g., `import` stuff)



## References
Guillaume, Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, and Chris Dyer. 2016. "[Neural Architectures for Named Entity Recognition](http://www.anthology.aclweb.org/N/N16/N16-1030.pdf)." *In Proceedings of NAACL-HLT*.

Jason P. C. Chiu and Eric Nichols. 2016. "[Named Entity Recognition with Bidirectional LSTM-CNNs](https://aclweb.org/anthology/Q16-1026)" *In TACL*.

Abbas Ghaddar and Phillippe Langlais. 2018. "[Robust Lexical Features for Improved Neural Network Named Entity Recognition](https://aclweb.org/anthology/C18-1161)" *In Proceedings of COLING*

Peng-Hsuan Li, Ruo-Ping Dong, Yu-Siang Wang, Ju-Chieh Chou, and Wei-yun Ma. 2017. "[Leveraging Linguistic Structures for Named Entity Recognition](https://aclweb.org/anthology/D17-1282)" *In Proceedings of EMNLP*
