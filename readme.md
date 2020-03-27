# Extracting Medical Entities from Social Media


![Model Diagram](fig/bwMODELNNNNbilstemcrf.jpg?raw=true "Model")

## Repostiory for the paper [Extracting Medical Entities from Social MEdia](https://dl.acm.org/doi/abs/10.1145/3368555.3384467)


## Requirements
```bash
numpy
pandas
seaborn
flair
tqdm
spacy
xgboost
sklearn
```

In this project, we implemented a deep learning method for medical entity extraction (in the natural language processing terminology (NLP) -- this is called named entity recognition (NER)) from social media text. The method is based on the BiLSTM+CRF architecture using RoBERTa contextual embeddings in combination with GloVe word embeddings.

We also created a novel labelled dataset for medical entity extraction called MedRed (from Reddit). Then we evaluated the method on two existing datasets: CADEC (from AskAPatient) and Micromed (from Twitter), a well as on MedRed (from Reddit). 

Finally, to validate the method on a large scale, we applied it on half a million Reddit posts from disease-specific subreddits (such as r/psoriasis and r/bpd). Then we studied how well the disease topic of each post can be predicted solely from the extracted entities.


## Structure

1. code
  * `train` contains the scripts to create the flair corpus from a given dataset and the for running the training models.
  * `evaluation` contains the scripts to evaluate the trained models on each of the 3 labelled datasets.
  * `validation` contains the scripts for applying the trained models on other datasets, and for disease prediction on Reddit from the extracted posts.

2. data `MedRed and Reddit can be downloaded from [FigShare](https://doi.org/10.6084/m9.figshare.12039609.v1). Others (i.e., CADEC and Micomed) are avilable from respective publications.`
  * Reddit
  * CADEC
  * Micromed
  * MedRed

3. preprocessed `the resulting pretrained models can also be found on FigShare.`
  * model
  
4. results `running the scripts will save the results in these folders.`
  * NER_res
   * CADEC
   * Micromed
   * MedRed 
   * Reddit  


We use [Flair library](https://github.com/flairNLP/flair) by Zalando Research.


## Licence

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details