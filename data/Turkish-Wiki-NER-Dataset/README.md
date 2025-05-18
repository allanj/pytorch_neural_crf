NER dataset from Wikipedia sentences. 20.000 sentences are sampled and re-annotated from [Kuzgunlar NER dataset](https://data.mendeley.com/datasets/cdcztymf4k/1).


Data split:

- 18.000 train
- 1000 test
- 1000 dev

Labels:

- CARDINAL
- DATE
- EVENT
- FAC
- GPE
- LANGUAGE
- LAW
- LOC
- MONEY
- NORP
- ORDINAL
- ORG
- PERCENT
- PERSON
- PRODUCT
- QUANTITY
- TIME
- TITLE
- WORK_OF_ART

Dataset is in **conll** format. Here's an example from the sample for you:

```
Kuyucak	B-GPE
batısında	O
Nazilli	B-GPE
ilçesi	O
,	O
doğusunda	O
Buharkent	B-GPE
ilçesiyle	O
çevrilidir	O
.	O
```

Annotations are done by [Co-one](https://co-one.co/). Many thanks to them for their contributions. This dataset is also used in our brand new spaCy Turkish packages.
If you wanna cite, kindly use:

[![DOI](https://zenodo.org/badge/558985121.svg)](https://zenodo.org/badge/latestdoi/558985121)

