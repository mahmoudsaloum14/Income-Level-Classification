# Income Level Classification Using Machine Learning

## Project overview

This project predicts whether a person's annual income is above $50,000 using demographic and employment information from the UCI Adult dataset. Its main objective is to compare three classifiers implemented directly with NumPy against a scikit-learn Random Forest while following a leakage-safe evaluation process.

## Dataset

The project uses the [UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult), also known as the Census Income dataset. The local `adult.csv` file contains 32,561 records and 14 input attributes. The binary target is `income` (`<=50K` or `>50K`). The data includes numeric variables such as age, education level, capital gain, and weekly working hours, as well as categorical variables such as occupation, work class, and marital status.

## Technologies

- Python 3.11
- NumPy and pandas
- scikit-learn for preprocessing, data splitting, metrics, cross-validation utilities, and Random Forest
- Matplotlib for exploratory and evaluation plots
- Jupyter Notebook for the documented workflow

## Preprocessing and feature selection

The workflow converts `?` values to missing values and removes 24 duplicate rows. Missing numeric values are replaced with the training median, while missing categorical values are replaced with the most frequent training category. Numeric features are standardized and categorical features are one-hot encoded. The text `education` column is removed because `education_num` expresses the same information in an ordered numeric form.

All preprocessing is fitted only on the training data. The cleaned data is divided into 80% training and 20% testing data with stratification and `random_state=42`. The test data is not used during tuning.

## Models

- Decision Tree implemented from scratch with NumPy using Gini impurity and greedy binary splits
- K-Nearest Neighbors implemented from scratch with NumPy using Euclidean distance and majority voting
- Perceptron implemented from scratch with NumPy using shuffled online weight updates
- Random Forest from scikit-learn

## Training and evaluation

A limited grid search uses three-fold stratified cross-validation on a reproducible 6,000-row subset of the training data. This keeps the from-scratch algorithms practical on an ordinary computer. The Random Forest search also compares standard training with balanced class weights. Balanced weights give the less frequent `>50K` class more influence during training and can improve its recall. The selected configuration is then trained on the full training set. Every final metric is calculated on the same untouched test set. F1-score is the main selection measure because the income classes are imbalanced; accuracy, precision, and recall are also reported.

Selected settings were: Decision Tree depth 5 with a minimum split size of 20; KNN with 15 neighbors; Perceptron learning rate 0.001 for 25 epochs; and Random Forest with 150 trees, no depth limit, a minimum leaf size of 2, and balanced class weights.

## Final results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest | 0.8333 | 0.6225 | 0.7825 | 0.6934 |
| KNN | 0.8439 | 0.7035 | 0.6084 | 0.6525 |
| Decision Tree | 0.8499 | 0.7724 | 0.5344 | 0.6317 |
| Perceptron | 0.7835 | 0.5361 | 0.7532 | 0.6264 |

Random Forest produced the highest test F1-score (0.6934), with a test accuracy of 0.8333. Balanced class weights increased its recall for the `>50K` class from 0.5957 to 0.7825 and raised F1-score from 0.6727 to 0.6934. This improvement comes with a lower precision and accuracy, which is an expected trade-off when the minority class receives more importance.

## Running the project

Place `adult.csv` in the project directory, then run:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook income_classification.ipynb
```

The complete experiment can also be run without Jupyter:

```bash
python run_experiment.py
```

Generated tables and plots are stored in `results/`. Running the script replaces these outputs with results from the current environment and data file.

## Possible improvements

Future work could test class weighting or resampling, tune classification thresholds, use a larger cross-validation sample, and compare additional ensemble methods. The custom tree could also be extended with pruning, while KNN could use a faster nearest-neighbor search for larger datasets.
