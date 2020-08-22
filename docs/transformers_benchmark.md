### Benchmark Performance (Cont)

We directly use BERT/Roberta-CRF for the following benchmark experiments. 
We strictly follow the optimizer configuration as in HuggingFace and use batch size 30.

* Experiments on the CoNLL-2003 dataset

    | Model| Dataset | Precision | Recall | F1 |
    |-------| ------- | :---------: | :------: | :--: |
    |HuggingFace Default (bert-base-cased)| Test Set | 90.71 | 92.04| 91.37|
    |HuggingFace Default (roberta-base)*| Test Set | 89.41 | 91.47|90.43|
    |BERT-base-cased (this repo)| Test set | 91.69 | 92.05 | 91.87 |
    |BERT-large-cased (this repo)| Test Set | 92.03 | 92.17 | 92.10 |
    |Roberta-base (this repo)| Test Set | 91.88  | 93.01 |92.44|
    |Roberta-large (this repo)| Test Set | **92.27**  | **93.18** |**92.72**|
HuggingFace Default (roberta-base)* has an issue with tokenization (There is no leading space).

We didn't achieve 92.4 F1 as reported in the BERT paper. 
I think one of the main reasons is they are using the document-level dataset instead of sentence-based dataset, and we didn't tune the hyperparameter such as optimizer yet. We use the same configuration as in HuggingFace.
 

* Experiments on the OntoNotes 5.0 dataset

    The benchmark performance for now:
    
    | Model| Dataset | Precision | Recall | F1 |
    |-------| ------- | :---------: | :------: | :--: |
    |BERT-base-cased (this repo)| Dev Set |  |  | |
    |BERT-large-cased (this repo)| Test Set | - | -|-|
    |Roberta-base (this repo)*| Test Set | 89.90  | 90.41 |**90.16**|
    
Roberta-base (this repo)* is still running. The others are not finished yet.