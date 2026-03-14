supercomputer access
będzie trenowanie

# Task 1 - Chemical ontology classification

ontology - hierarchy of entities
dog -> canine -> mammal -> animal

hierarchical classification -> classify a molecule to a class and its all parent classes

input SMILES, molecular graphs, need to vectorize this data
moelcular fingerptins, GNNs, graph/SMILES transformers
scikit-fingerprints

We get:
Train set (33k molecules, 500 classes)
Class graph (DAG)
test set: (11k classes)

submit .parquet with classifications and a link to repo
metrics: F1-score, macro-averaged mean (mean of F1 scores from all classes)
then: graph consistency metric

There is a public leaderboard for some half of molecules

# Task 2 - context collection strategy

code completion basically across whole repo

evaluation: comparison to they're models - ChrF score


# Task 3 - time series

house energy data every 5 minutes, no data from summer, predict for summer

2 csv files - logs from devices and something

prediction: for every device for every month predict temperature

evaluation: MAE

time series, data out of distribution

maybe plot some temperatures cause we know them xd


# Task 4 - ECG digitization

ocr
train 3k images
test 500 images

easy, medium, hard

evaluation:
signal shape - Pearson coefficient, Amplitude
time calibration