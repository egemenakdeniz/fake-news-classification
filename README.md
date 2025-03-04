# Fake News Classification with BERT

This project focuses on classifying fake and real news using the **FakeNewsNet** dataset and **BERT** (Bidirectional Encoder Representations from Transformers). The model fine-tunes the pre-trained BERT model for binary classification.

## Dataset

The dataset used in this project is **FakeNewsNet**, which consists of labeled fake and real news articles. You can download it from Kaggle:

[FakeNewsNet Dataset]([www.kaggle.com/datasets/mdepak/fakenewsnet](https://www.kaggle.com/datasets/mdepak/fakenewsnet))

### Data Structure
The dataset contains two sources:
- **BuzzFeed News** (Fake & Real)
- **PolitiFact News** (Fake & Real)

Each CSV file consists of news content with labels:
- `1` → Fake News
- `0` → Real News

## Requirements

Ensure you have the following dependencies installed:

```bash
pip install pandas torch transformers scikit-learn
```

## Model Training

The script follows these steps:

1. **Load the dataset**: Reads the CSV files and assigns labels.
2. **Preprocessing**: Tokenizes the news titles using BERT Tokenizer.
3. **Dataset Preparation**: Converts text into tensors and splits into train-test sets.
4. **Fine-Tuning BERT**: Uses a pre-trained BERT model for binary classification.
5. **Evaluation**: Computes accuracy and classification metrics.

## Usage

Run the script to train the model:

```bash
fake-news-classification.py
```

## Model Performance

The model is trained using **3 epochs** with `AdamW` optimizer and `CrossEntropyLoss`. You can modify the `epochs` parameter in `train_model()` to experiment with different training durations.

After training, the script evaluates the model and prints classification metrics (accuracy, precision, recall, and F1-score).

## Contributing
If you have any suggestions, improvements, or find any issues, feel free to open an issue or submit a pull request. Contributions are welcome!


