# Readme file for the project Extracting Medical Entities from Social Media


## Requirements
```bash
numpy
pandas
flair

```

In this project, we implemented deep learning a method for entity extraction, or, in 
natural language processing temrinology, for named entitiy recognition. The method  is
based on BiLSTM+CRF architecture using contextual RoBERTa embeddings.

We also created in this work a novel labelled dataset for medical entities work called MedRed (Reddit).
Then we evaluated the method on two existing datasets: CADEC (AskAPatient) and Micromed (Twitter), a
well as on MedRed. 

Finally, to validate the method on a large scale, we applied it on half a million
Reddit posts from specific disease subreddits (such as r/psoriasis and r/bpd). Then we
studied how well the disease topic of each post can be predicted solely using the extracted entities.


## Structure

1. code
  * train
  * evaluation
  * validation

2. data
  * Reddit
  * CADEC
  * Micromed
  * MedRed

3. preprocessed
  * model
4. results
  * NER_res
   * CADEC
   * Micromed
   * MedRed 
   * Reddit  


## NER

We use Flair libarary by Zalando Research.

* `train` contains the scripts to create the flair corpus from a give dataset and the for running the method.

* `evaluation` contains the scripts to evaluate results of the method on each labelled dataset.

* `validation` contains the scripts for disease prediciton on Reddit from the extracted posts.

## Licence