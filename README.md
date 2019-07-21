## LSTM-CRF Model for Named Entity Recognition

This repository implements an LSTM-CRF model for named entity recognition. The model is same as the one by [Lample et al., (2016)](http://www.anthology.aclweb.org/N/N16/N16-1030.pdf) except we do not have the last `tanh` layer after the BiLSTM.

### Requirements
* Python 3.6
* Tested on PyTorch >=0.4.1


### Usage
1. Put the Glove embedding file (`glove.6B.100d.txt`) under `data` directory (You can also use ELMo/BERT/Flair, Check below.) Note that if your embedding file does not exist, we just randomly initalize the embeddings.
2. Simply run the following command and you can obtain results comparable to the benchmark above.
    ```bash
    python3.6 main.py
    ```
### Running with your own data. 
1. Create a folder `YourData` under the data directory. 
2. Put the `train.txt`, `dev.txt` and `test.txt` files (make sure the format is compatible) under this directory. 
If you have a different format, simply modify the reader in `config/reader.py`.
3. Change the `dataset` argument to `YourData` in the `main.py`.
  
    
### Using ELMo (and BERT)
There are two ways to import the ELMo and BERT representations. We can either __preprocess the input files into vectors and load them in the program__ or __use the ELMo/BERT model to _forward_ the input tokens everytime__. The latter approach allows us to fine tune the parameters in ELMo and BERT. But the memory consumption is pretty high. For the purpose of most practical use case, I simply implemented the first method.
1. Run the scripts under `preprocess/get_elmo_vec.py`. As a result, you get the vector files for your datasets.
2. Run the main file with command: `python3.6 main.py --context_emb elmo`. You are good to go.

For using BERT, it would be a similar manner. Let me know if you want further functionality. Note that, we concatenate ELMo and word embeddings (i.e., Glove) in our model (check [here](https://github.com/allanj/pytorch_lstmcrf/blob/master/model/lstmcrf.py#L82)). You may not need concatenation for BERT.
    

### Benchmark Performance

Empirically, although `ADAM` optimizer converges faster, we found that using `SGD` with learning rate of `0.01` and `100` epochs is better than `ADAM`.

* Experiments on the CoNLL-2003 dataset
    
    | Model| Dataset | Precision | Recall | F1 |
    |-------| ------- | :---------: | :------: | :--: |
    |Lample et al., (2016)| Dev Set | - | -|-|
    |This Implementation (on GPU)| Dev Set | 94.99 | 94.85 |94.92|
    |Lample et al., (2016)| Test Set | - | -|90.94|
    |This Implementation (on GPU)| Test Set | 91.30  | 91.41 |**91.36**|
    | This Implementation +ELMo| Test Set | 92.4  | 92.2 |**92.3**|

* Experiments on the OntoNotes 5.0 dataset
    Since the dataset statistics is usually not clear in many literatures. We take a lot efforts to find the standard splits in the following table. 
    
    |  | #Sent | #Entity | #Token | 
    |---|:----:|:----:|:----:|
    |Train|59,924 | 81,828|1,088,503|
    |Dev|8,528 | 11,066|147,724|
    |Test|8,262 | 11,257|152,728|
    
    The above statistics follow most of the paper that have dataset statistics table presented (Chiu and Nichols, 2016; Li et al., 2017; Ghaddar and Langlais, 2018;). 

    **Dataset Preprocessing**: We found that most of the papers are not describing the data splits in details and there are two different ways to create the dataset.
    The preprocessing scripts can be either found in http://conll.cemantix.org/2012/data.html or http://cemantix.org/data/ontonotes.html.
    However, one problem is no matter how you preprocess the data using either one of this scripts, you could not get the exact data splits as in the above table (i.e., you
    could not obtain the exact splits as many literatures).

    **How to get the correct splits?** We found that we should use the train/dev splits with the preprocessing scripts from http://conll.cemantix.org/2012/data.html and use the test split with the
    preprocessing scripts from http://cemantix.org/data/ontonotes.html. Then you will obtain the above exact data splits.

    The benchmark performance (without contextualized embeddings):
    
    | Model| Dataset | Precision | Recall | F1 |
    |-------| ------- | :---------: | :------: | :--: |
    |LSTM-CNN (Chiu and Nichols, 2016)| Test Set | - | -|86.17|
    |BiLSTM-CRF (Our Implementation on GPU)| Test Set | 87.85 | 86.84 |**87.34**|
    | Our Implementation  +ELMo| Test Set | 89.14| 88.59 |**88.87**|
    |LSTM-CNN + lexicon (Chiu and Nichols, 2016)*| Test Set | - | -|86.28|
    |BRNN-CNN with parse tree (Li et al., 2017)*| Test Set | 88.0 | 86.5|87.21| 
    |BiLSTM-CRF + Robust Features (Ghaddar and Langlais, 2018)*| Test Set | - | -|87.95| 
    
    \* indicates they use external features besides word embeddings. 
    The results can be reproduced by simply changing the dataset from `conll2003` to `ontonotes`.


### Ongoing Plan

- [x] Support for ELMo as features
- [ ] Interactive model where we can just import model and decode a setence
- [ ] Make the code more modularized (separate the encoder and inference layers) and readable (by adding more comments)
- [ ] Put the benchmark performance documentation to another markdown file
- [ ] Integrate ELMo/BERT as a module instead of just features.
- [ ] Clean up the code to better organization (e.g., `import` stuff)



## References
Guillaume, Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, and Chris Dyer. 2016. "[Neural Architectures for Named Entity Recognition](http://www.anthology.aclweb.org/N/N16/N16-1030.pdf)." *In Proceedings of NAACL-HLT*.

Jason P. C. Chiu and Eric Nichols. 2016. "[Named Entity Recognition with Bidirectional LSTM-CNNs](https://aclweb.org/anthology/Q16-1026)" *In TACL*.

Abbas Ghaddar and Phillippe Langlais. 2018. "[Robust Lexical Features for Improved Neural Network Named Entity Recognition](https://aclweb.org/anthology/C18-1161)" *In Proceedings of COLING*

Peng-Hsuan Li, Ruo-Ping Dong, Yu-Siang Wang, Ju-Chieh Chou, and Wei-yun Ma. 2017. "[Leveraging Linguistic Structures for Named Entity Recognition](https://aclweb.org/anthology/D17-1282)" *In Proceedings of EMNLP*
