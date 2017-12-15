# Logistic Regression using Numpy

This project is the second part of my will to understand and implement Machine Learning. My objective is to understand the key concepts firts and then start using Frameworks for it.

In the first step of this journey, I implemented the Linear Regression Model, the second part is to implement and understand the Logistic Regression Model.

### Logistic Regression Basics

Another widely used Algorithm and a great performer with some regression tasks and predict new outputs based on the relationship of the prior examples in the training set. A real world analysis can be made with a dataset representing microchips tests from a factory, in the plotted data is possible to see that there is a way to distinguish between the working and the defect microchips in a linear way.

Looking at the whole picture, the Logistic Regression does not have a lot o differences from the previously described Linear Regression, there is a Feedforward step that makes the predictions and calculate the Loss and there is the Backpropagation step where the error is minimized and the Weights and bias are updated with the calculated Gradients. Going a little more in depth in the Feedforward in Logistic Regression, there is one more step in this part, the linear function now transforms into a non-linear function by the Activation function. The most common Activation function used in Machine Learning is called Sigmoid and it takes the prediction output from the linear computation and makes it non-linear. The Loss function must change as well, the error must be calculated in a non-linear form.

Although the Backpropagation step remains necessary it must change as well, since the Loss function changed, the Gradient Descent must minimize another function, so the partial derivatives are not the same. The weights and bias must be updated.

The added non-linearity changes a lot in terms of code and in terms of math from Linear Regression to Logistic Regression. It gives more precision, the output of the algorithm can be a Boolean analysis just like the example, Good Microchip or Defective Microchip, and depending of its value the probability of each case can be recovered. This was just a glimpse of what Logistic Regression can be and how it functions, but it can be a great introduction to a more complicated type like Neural Networks.

### Prerequisites

To keep things simple, this project will use just two libraries in python, one to make math computations and other to plot the graph with the dataset and the Line of Best Fit.

So in yout Python3 enviroment just make sure that you have this dependencies installed.

```
pip install numpy
```
```
python -mpip install -U pip
python -mpip install -U matplotlib
```

### Running

To run this project just make sure that both, python file is in the same location as the Microchip_testing.txt file

To run it just execute the following command line:
```
python3 LogisticRegressionNumpy.py
```

## Results

The Hyperparameters are initialized with zeros, so the final total cost may be close to 0.3612..

The Curve that differentiates the data must be plotted dynamically with its change over time with the dataset behind it, so we can see the learning process happening.


## References

This project was heavily influenced by the Coursera Courses of Andrew NG, Machine Learning and Deep Learning Specialization. There are concepts applied here of both courses and the example and dataset is from the Machine Learning Course.

Machine Learning:
https://www.coursera.org/learn/machine-learning

Deep Learning Specialization:
https://www.coursera.org/specializations/deep-learning

Thanks ;D