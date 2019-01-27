# human-protein-atlas-kaggle-competition
This repository contains my solution which was Ranked in [Top-17%](https://www.kaggle.com/disisbig/competitions) in the [final leaderboard](https://www.kaggle.com/c/human-protein-atlas-image-classification/leaderboard) in [Human Protein Atlas Image Classification](https://www.kaggle.com/c/human-protein-atlas-image-classification) challenge on Kaggle.

The challenge was to develop models capable of classifying mixed patterns of proteins in microscope images.


## My Solution

My solution used BnInception model as its backbone, used weighted loss to handle unbalanced classes. The final solution was an ensemble
of various models which I had developed during the course of challenge.

## Things I tried:

* SMOTE for class-imbalance problem

* Ensemble of models of varying architectures, varying input size and varying optimizers.

* Different Loss functions like BCE, F1 Loss, Focal Loss etc

* Adding/Removing HPA data

## What worked for me

* Weighted loss gave me good results consistently, as compared to Oversampling

* Ensemble of similar models didn't help performance

* `BCEWithLogitsLoss` worked best for me
