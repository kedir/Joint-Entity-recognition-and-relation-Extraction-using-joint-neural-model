# Joint entity recognition and relation extraction as a multi-head selection problem

Implementation of the papers
[Joint entity recognition and relation extraction as a multi-head selection problem](https://arxiv.org/abs/1804.07847)

# Requirements
* Numpy 1.17.1
* Gensim 3.4.0
* Tensorflow 1.13.1
* PrettyTable 0.7.2
* Pandas 1.1.5

## Task
Given a sequence of tokens (i.e., sentence), (i) give the entity tag of each word (e.g., NER) and (ii) the relations between the entities in the sentence. The following example indicates the accepted input format of our multi-head selection model:


```
0	Marc		B-PER		['N']					[0]		
1	Smith		I-PER 		['lives_in','works_for']		[5,11]
2 	lives		O		['N']					[2]
3	in		O		['N']					[3]
4	New		B-LOC		['N']					[4]
5	Orleans		I-LOC		['N']					[5] 
6	and		O		['N']					[6]
7	is		O		['N']					[7]
8	hired		O		['N']					[8]
9	by		O		['N']					[9]
10	the		O		['N']					[10]
11  government		B-ORG		['N']					[11]
12	.		O		['N']					[12]
```

## Configuration
The model has several parameters such as: 
* EC (entity classification) or BIO (BIO encoding scheme)
* Character embeddings
* Ner loss (softmax or CRF)

that could be specified in the configuration files (see [config](https://github.com/bekou/multihead_joint_entity_relation_extraction/tree/master/configs)).

## Run the model

In order to run the model use the notebook named as " multi_head_joint_entity_relation_extraction.ipynb "
and run on google colab or just follow the instruction below:

```
./run.sh
```

## More details
Commands executed in ```./run.sh```:

1. Train on the training set and evaluate on the dev set to obtain early stopping epoch
```python3 train_es.py```

## Notes

Please cite the authors work(paper) when using this software.

Giannis Bekoulis, Johannes Deleu, Thomas Demeester, Chris Develder. Joint entity recognition and relation extraction as a multi-head selection problem. Expert Systems with Applications, Volume 114, Pages 34-45, 2018