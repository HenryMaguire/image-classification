# Introduction

This is an assignment for the Udacity Machine Learning Nanodegree program. It is an image classifier based on convolutional neural networks using the [cifar-10](https://www.kaggle.com/c/cifar-10) dataset. It is implemented using Tensorflow in Python 2.7.

*** Note: this project is still a work in progess. ***

## System requirements
- Numpy
- Scipy
- Tensorflow
- Scikit-learn (v0.17)

## Getting started

- Open the Jupyter/IPython notebook in Python 2.7
- Download the data within the notebook and follow the cells/notes.

## Notes
The original notebook was written in Python 3 by Udacity. Since the `helper.py` and `problem_unittests.py` were provided by Udacity also, these originally contained some features not present in Python 2.7:
- In the `load` in-built function, there is an extra argument of `encoding='Latin1'` which is not valid in Python 2.7. I just removed this.
- In the Python 3 `unittest` module there is something called `MagicMock()` which I do not understand and is not a feature in Python 2.7. I just deleted this and ceased using this unittest in the main IPython Notebook.


## Todo

- Run the code on a better computer! Maybe on an AWS instance.
