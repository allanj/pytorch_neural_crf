### Running with your own data. 
1. Create a folder `YourData` under the data directory. 
2. Put the `train.txt`, `dev.txt` and `test.txt` files (make sure the format is compatible) under this directory. 
If you have a different format, simply modify the reader in `config/reader.py`.
3. Change the `dataset` argument to `YourData` in the `main.py`.
  
