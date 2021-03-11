
### Fast CRF Usage

Our implementation is inspired from the papers by Rush (2020).
1. Move from Linear-CRF to Fast Linear-CRF. 

For example, in `src/model/transformers_nerualcrf.py`
```python 
#import the module
from src.model.module.fast_linear_crf_inferencer import FastLinearCRF

#replace self.inferencer = LinearCRF in the init file.  
self.inferencer = FastLinearCRF(...)
```

2. Revise evaluation in `src/config/eval.py`.

Remove the following line 54:
```python
prediction = prediction[::-1] 
```

The reason of doing this is because the viterbi from `FastLinearCRF` already produce the standard order sequence, we do not have to do the reverse.




### References
Alexander Rush, 2020, [Torch-Struct: Deep Structured Prediction Library](https://arxiv.org/abs/2002.00876), [Github](https://github.com/harvardnlp/pytorch-struct)

Simo Särkkä, Ángel F. García-Fernández. 2020, [Temporal Parallelization of Bayesian Smoothers](https://arxiv.org/abs/1905.13002)

Guy E. Blelloch, 1993, [Prefix Sums and Their Applications](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)
