# README

This README provides an overview of the Wine Origin Prediction using Artificial Neural Networks implemented in Python for the given dataset(wine.data).

### Group Number

- 90

### Roll Numbers

- 21ME30078 Debraj Das

### Project Number

- WONN

### Project Title

- Wine Origin Prediction using Artificial Neural Networks

## Setup

To run the code successfully, follow these steps:

1. Clone the repository or download the `wine.data` dataset.
2. Ensure the dataset file (`wine.data`) is in the same directory as the Python script.
3. Install the required dependencies using pip by running:
   ```
   pip install -r requirements.txt
   ```
4. Execute the Python script (`90_WONN.py`).
   ```
   python 90_WONN.py
   ```

## Requirements

You can find the list of required dependencies in the `requirements.txt` file. Install them using the command mentioned above.

## How to Run the Code

The script will load the dataset, preprocess it, train the custom Artificial Neural Networks, and compare its performance with pytorch's Artificial Neural Networks.

## Methodology

1. **Data Loading and Preprocessing**:

   - The dataset (`wine.data`) is loaded using Pandas.
   - Categorical variables are converted to numerical using label encoding.

2. **Custom Artificial Neural Networks**:

   - A custom Artificial Neural Networks is implemented from scratch.
   - It calculates class probabilities and feature probabilities using Forward passing and Backward passing.

3. **pytorch's Artificial Neural Networks**:

   - The dataset is split into training and testing sets using Scikit-learn's `train_test_split`.
   - pytorch's Artificial Neural Networks is trained and tested.

4. **Performance Comparison**:
   - The accuracy and classification report are printed for both custom and Scikit-learn's Naive Bayes Classifiers.

## Insights

- The code demonstrates the implementation of a Artificial Neural Networks from scratch and compares its performance with pytorch's Artificial Neural Networks.
- It provides insights into the effectiveness of the custom implementation compared to the library-provided solution.

## Additional Notes

- Both Artificial Neural Networks are trained on the same dataset and evaluated using the same test set for fair comparison.
- Experimentation with hyperparameter tuning can further optimize the performance of both Neural Networks.

## Support

For any questions or issues, please contact

- `susmita834805@gmail.com`
- `debrajdas.kgpian.iitkgp.ac.in`.
