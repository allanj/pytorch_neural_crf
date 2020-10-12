
### Usage with Word Embedding (e.g., Glove)
1. Put the Glove embedding file (`glove.6B.100d.txt`) under `data` directory (You can also use ELMo/BERT/Flair, Check below.) Note that if your embedding file does not exist, we just randomly initalize the embeddings.
2. Simply run the following command and you can obtain results comparable to the benchmark above.
    ```bash
    python trainer.py
    ```
    If you want to use your 1st GPU device `cuda:0` and train models for your own dataset with elmo embedding:
    ```
    python trainer.py --device=cuda:0 --dataset=YourData --model_folder=saved_models 
    ```



### Using ELMo (and BERT) as contextualized word embeddings (Static, Feature-based Approach, with Externally Stored Vector)
There are two ways to import the ELMo and BERT representations. We can either __preprocess the input files into vectors and load them in the program__ or __use the ELMo/BERT model to _forward_ the input tokens everytime__. The latter approach allows us to fine tune the parameters in ELMo and BERT. But the memory consumption is pretty high. For the purpose of most practical use case, I simply implemented the first method.
1. Run the script with `python -m preprocess.get_elmo_vec YourData`. As a result, you get the vector files for your datasets.
2. Run the main file with command: `python trainer.py --static_context_emb elmo`. You are good to go.

For using BERT, it would be a similar manner. Let me know if you want further functionality. Note that, we concatenate ELMo and word embeddings (i.e., Glove) in our model (check [here](https://github.com/allanj/pytorch_lstmcrf/blob/master/model/bilstm_encoder.py#L67)). You may not need concatenation for BERT.
You need to install [HuggingFace Transformers](https://github.com/huggingface/transformers) and [BERT-As-Service](https://github.com/hanxiao/bert-as-service) before running the following preprocessing script.
1.  Run the script with `python -m preprocess.get_bert_vec YourData`.

I suggest you also quickly read the documentation in BERT-As-Service before preprocessing.
