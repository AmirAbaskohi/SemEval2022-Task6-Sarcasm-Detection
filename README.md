# UTNLP at SemEval-2022 Task 6: A Comparative Analysis of Sarcasm Detection Using Generative-based and Mutation-based Data Augmentation

## Abstract
Sarcasm is a term that refers to the use of words to mock, irritate, or amuse someone. It is commonly used on social media. The metaphorical and creative nature of sarcasm presents a significant difficulty for sentiment analysis systems based on affective computing. The technique and results of our team, UTNLP, in the SemEval-2022 shared task 6 on sarcasm detection are presented in this paper. We put different models, and data augmentation approaches to the test and report on which one works best. The tests begin with fundamental machine learning models and progress to transformer-based and attention-based models. We employed data augmentation based on data mutation and data generation. Using RoBERTa and mutation-based data augmentation, our best approach achieved an F1-score of 0.38 in the competition’s evaluation phase. After the competition, we fixed our model’s flaws and achieved an F1-score of 0.414.

## Models
All the models are available in `Models` directory. In addition to models reported in the paper, we have implented `XLNet` and `Electra` models, which are available in in the models directory.

To run the models, first make sure to install the requirements using below command:
```
pip install -r requirements.txt
```
Then run each models by using calling `python` for the model name like:
```
python ./Models/model-name/model-name.py
```

## Data
We mostly used the `Isarcasm (Oprea and Magdy, 2019)` dataset in this study. In specific experiments, we integrated the primary dataset with various secondary datasets, including the `Sarcasm Headlines Dataset (Misra and Arora, 2019)` and `Sentiment140 dataset (Go et al., 2009)` to increase the quantity of data and compensate for the lack of sarcastic data.

Data directory includes our dataset which are in the following fomat:
```
|
____ Data
    |
    ___ Cleaned Dataset/
    |
    ___ Foreign Dataset/
    |
    ___ Main Dataset/
    |
    ___ Mutant Dataset/
    |
    ___ Train_Dataset.csv
    |
    ___ Test_Dataset.csv
```

`Cleaned Dataset` includes main dataset with preprocessing. `Foregin Dataset` includes our secondary datasets like `Sentiment 140`, they available in <a href="https://drive.google.com/drive/folders/1NSXGPRQnuSP2ipNG6-I-7FF-tR9iZvVE?usp=sharing">this link</a>. `Main Dataset` is the task train dataset. `Mutant Dataset` is our mutated version of main task dataset. Each dataset name has a three bit in it. First bit shows using word elimination. Second bit is for word replacement and third one is for shuffling. `Train_Dataset.csv` is the concatation of main task dataset and `mutant100` which we find it the best dataset for the models. `Test_Dataset.csv` is the task test dataset with labels.

Here is the dataset mapping:

| **File Name** |          **Dataset**         |
|:-------------:|:----------------------------:|
|     train1    | Sarcasm Headlines Dataset V1 |
|     train2    | Sarcasm Headlines Dataset V2 |
|     train3    |     Twitter News Dataset     |
|     train4    |     Sentiment 140 Dataset    |
|     train5    |             SPIRS            |
|     train6    | Twitter US Airline Sentiment |

## Data Augmentation
In this directoy, we have our data augmentor models.

`Generative Pre-trained Transformer 2 (GPT-2)` is an open-source artificial intelligence created by OpenAI in February 2019. GPT-2 translates text, answers questions, summarizes passages, and generates text output on a level that, while sometimes indistinguishable from that of humans, can become repetitive or nonsensical when generating long passages. It is a general-purpose learner; it was not specifically trained to do any of these tasks, and its ability to perform them is an extension of its general ability to accurately synthesize the next item in an arbitrary sequence. GPT-2 was created as a "direct scale-up" of OpenAI's 2018 GPT model, with a ten-fold increase in both its parameter count and the size of its training dataset.

![image](https://user-images.githubusercontent.com/50926437/155897747-876044f4-7960-4787-8cac-459facb3b80a.png)

We used three distinct ways to change the data in `Mutation-based` data augmentation: eliminating, replacing with synonyms, and shuffling. These processes were used in the following order: shuffling, deleting, and replacing. The removal and replacement were carried out systematically. We used the words’ roots to create a synonym dictionary. When a term was chosen to be swapped with its synonyms, we chose one of the synonyms uniformly at random(Figure 1). We tried each combination of these processes to find the best data augmentation combination (a total of seven).

## Data Preprcoessing
Here is the code which we used for preprocessing. We have checked different methods like Stemming, Lemmaitization, Link removation, and etc.

## Paper
Includes our task paper. For more information, we suggest you to read this paper.

## Results

F1-score and accuracy for different data augmentation
methods on SVM model with BERT word
embedding and no preprocessing.

| Data Augmentation      | F1-Score | Accuracy     |
| :---        |    :----:   |          ---: |
| Shuffling      | 0.305       | 0.7471   |
| Shuffling + Replacing   | 0.3011        | 0.7414      |
| Shuffling + Elimination   | 0.3064        | 0.7478      |
| Elimination   | 0.301        | 0.7478      |
| GPT-2   | 0.2923        | 0.675      |

Best results for each model using iSarcasm
dataset and mutation-based data augmentation.

| Model      | F1-Score | Accuracy     |
| :---        |    :----:   |          ---: |
| SVM      | 0.3064       | 0.7478   |
| LSTM-based   | 0.2751        | 0.7251      |
| BERT-based   | 0.414        | 0.8634      |
| Attention-based   | 0.2959        | 0.7793      |
| Google’s T5   | 0.4038        | 0.8124      |
| Electra   | 0.2907        | 0.7642      |
| Google’s T5   | 0.3684        | 0.7918      |

## Acknowledgements
We want to convey our heartfelt gratitude to
Prof. Yadollah Yaghoobzadeh from the University
of Tehran, who provided us with invaluable advice
during our research. We would also like to thank
Ali Edalat from the University of Tehran, who provided
us with the initial proposals for resolving the
dataset’s imbalance problem.

## Cite
```bibtex
@inproceedings{abaskohi-etal-2022-utnlp,
    title = "{UTNLP} at {S}em{E}val-2022 Task 6: A Comparative Analysis of Sarcasm Detection using generative-based and mutation-based data augmentation",
    author = "Abaskohi, Amirhossein  and
      Rasouli, Arash  and
      Zeraati, Tanin  and
      Bahrak, Behnam",
    booktitle = "Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.semeval-1.135",
    pages = "962--969",
    abstract = "Sarcasm is a term that refers to the use of words to mock, irritate, or amuse someone. It is commonly used on social media. The metaphorical and creative nature of sarcasm presents a significant difficulty for sentiment analysis systems based on affective computing. The methodology and results of our team, UTNLP, in the SemEval-2022 shared task 6 on sarcasm detection are presented in this paper. We put different models, and data augmentation approaches to the test and report on which one works best. The tests begin with traditional machine learning models and progress to transformer-based and attention-based models. We employed data augmentation based on data mutation and data generation. Using RoBERTa and mutation-based data augmentation, our best approach achieved an F1-score of 0.38 in the competition{'}s evaluation phase. After the competition, we fixed our model{'}s flaws and achieved anF1-score of 0.414.",
}
```
