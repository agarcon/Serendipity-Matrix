# Serendipity Matrix 

The package provides the methods to provide the serendipity matrix and for visualizing the serendipity matrix in a horizontal bar chart. The serendipity matrix is an innovative method to understand the behavior of a prediction model for classification problems. 

This matrix provides information about the degree of certainty that the classifier has in its own predictions, indicating whether it is robust and reliable or uncertain and doubtful. This method has two variants: the class-independent serendipity
matrix and the class-specific serendipity matrix depending on the kind of analysis required. 

By analyzing the data provided by them, our goal is to improve the reliability and explainability of prediction models and to provide users with a clearer understanding of why a model is certain or uncertain about its predictions.

## Installation

Serendipity Matrix can be installed from [PyPI](https://pypi.org/project/serendipity_matrix/)

```bash
pip install serendipity_matrix
```

Or you can clone the repository and run:

```bash
pip install .
```

## Sample usage

```python
from serendipity_matrix import class_indep_matrix, class_spec_matrix, plot_class_spec
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits

# Loads the dataset
X, y = load_digits(return_X_y=True)

# Training and predict
model = RandomForestClassifier(random_state=0).fit(X, y)
result = model.predict_proba(X)

# Calculates and prints the class-independent serendipity matrix
ci_matrix = class_indep_matrix(y, result)
print(ci_matrix)

# Calculates and prints the class-specific serendipity matrix
cs_matrix = class_spec_matrix(y, result)
print(cs_matrix)

# Plots the class-specific serendipity matrix
plot_class_spec(y, result)
```

## Result sample

### Class-independent serendipity matrix

|Reliability|Overconfidence|Underconfidence|Serendipity|
|:---------:|:------------:|:-------------:|:---------:|
| 0.85208   |   0.134924   |    0.006851   |  0.006145 |

### Class-specific serendipity matrix

|CLASS_NAME|Reliability|Overconfidence|Underconfidence|Serendipity|
|:--------:|:---------:|:------------:|:-------------:|:---------:|
|    0     |  0.988850 |  0.010500    |    0.000650   |7.49955e-19|
|    1     |  0.822048 |  0.160756    |    0.009725   |7.47143e-03|
|    2     |  0.954939 |  0.017090    |    0.004263   |2.37083e-02|
|    3     |  0.936688 |  0.052884    |    0.002245   |8.18263e-03|
|    4     |  0.966523 |  0.018899    |    0.001481   |1.30976e-02|
|    5     |  0.910906 |  0.081227    |    0.004087   |3.78039e-03|
|    6     |  0.959340 |  0.036419    |    0.004240   |4.17679e-32|
|    7     |  0.721232 |  0.268712    |    0.010053   |3.59522e-06|
|    8     |  0.578063 |  0.398610    |    0.015763   |7.56317e-03|
|    9     |  0.926385 |  0.057183    |    0.009373   |7.05917e-03|

### Class-specific serendipity matrix horizontal bar chart

![Class-specific serendipity matrix](Resources/Example_class-specific_serendipity_matrix_for_digits_dataset.png)


## Citation

The methodology is described in detail in:

[1] J. S. Aguilar-Ruiz and A. García Conde, “”<!-- , Scientific Reports, 14:10759, 2024, doi: 10.1038/s41598-024-61365-z. Also, the mathematical background of the multiclass classification performance can be found in: in IEEE Access.-->