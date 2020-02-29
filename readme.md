# Extracting Medical Entities from Social Media


![Model Diagram](fig/bwMODELNNNNbilstemcrf.jpg?raw=true "Model")

## Requirements
```bash
numpy
pandas
flair

```

In this project, we implemented a deep learning method for medical entity extraction (in the natural language processing terminology (NLP) -- this is called named entity recognition (NER)) from social media text. The method is based on the BiLSTM+CRF architecture using contextual RoBERTa embeddings.

We also created in this work a novel labelled dataset for medical entities called MedRed (from Reddit). Then we evaluated the method on two existing datasets: CADEC (from AskAPatient) and Micromed (from Twitter), a well as on MedRed. 

Finally, to validate the method on a large scale, we applied it on half a million Reddit posts from disease-specific subreddits (such as r/psoriasis and r/bpd). Then we studied how well the disease topic of each post can be predicted solely from the extracted entities.


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


We use Flair library by Zalando Research.

* `train` contains the scripts to create the flair corpus from a given dataset and the for running the method.

* `evaluation` contains the scripts to evaluate results of the method on each labelled dataset.

* `validation` contains the scripts for disease prediction on Reddit from the extracted posts.

## Licence

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details