## LSTM-CRF Model for Named Entity Recognition

This repository implements an LSTM-CRF model for named entity recognition. The model is same as the one by [Lample et al., (2016)](http://www.anthology.aclweb.org/N/N16/N16-1030.pdf) except we do not have the last `tanh` layer after the BiLSTM.

### Requirements
* Python 3.6
* Tested on PyTorch 0.4.1

### Benchmark Performance

We conducted experiments (with GPU) on the CoNLL-2003 dataset.

| Model| Dataset | Precision | Recall | F1 score |
|-------| ------- | :---------: | :------: | :--: |
|Lample et al., (2016)| CoNLL-2003 | - | -|90.94|
|This Implementation| CoNLL-2003 | 91.30  | 91.41 |91.36|


### Usage
1. Put the Glove embedding file (`glove.6B.100d.txt`) under `data` directory
2. Create two folders: `model_files` and `results` to save the models and results, respectively.
3. Simply run the following command and you can obtain results comparable to the benchmark above.
    ```bash
    python3.6 main.py
    ```

### Using ELMo/BERT/Flair
1. Copy the vector files to the `data/conll-2003` folder.

## References
Lample, Guillaume, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, and Chris Dyer. "[Neural Architectures for Named Entity Recognition](http://www.anthology.aclweb.org/N/N16/N16-1030.pdf)." *In Proceedings of NAACL-HLT*, pp. 260-270. 2016.
