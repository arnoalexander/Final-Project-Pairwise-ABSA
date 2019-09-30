# Pairwise Aspect-Based Sentiment Analysis

Extracting pairs of aspect and sentiment expression from Indonesian hotel reviews. 
  
The approach is to identify all possible candidate pairs and then classify them into valid or invalid class.  

![](https://i.imgur.com/Emg584r.png "Example")

## Project Structure

```
.
├── ner
├── pairing
└── utility
```

* **ner** : Classify tokens whether they are aspect term, sentiment term, or not both. This is only a sandbox, we won't focus on it.
* **pairing** : Main module, containing file reader, feature extractor, and pair classifier.
* **utility** : Helper for main module.

## How to Run

Create directories to store data and save trained model. Put train and test dataset to `data/labelled`.  

```
.
├── data
|   ├── labelled
|   |   ├── train-pair.txt
|   |   └── test-pair.txt
|   └── raw
└── model
    ├── ner
    ├── pairing
    └── utility
```

Then, run `demo.ipynb` for demonstration or any other jupyter notebook files.
