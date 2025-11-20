# Softmax Regression

## From Regression to Classification
Regression is the tool we will go to when we want to answer *how much?* or *how many?*,
But even within regression models, there's a lot of distinctions to make based on the nature of the data.

For instance, some problems like the price of a house will never be negative, and changes relative to its baseline price.
So it might be effective to regress on the *logarithm* of the price rather than the raw value.

Likewise, the number of days a patient spends in a hospital is a *discrete nonnegative random variable*. 
In this case, least mean squares might not be the ideal approach.
This sort of time-event modeling are dealt with a subfield called *survival modeling*.

The point here is to just let you know that there is a lot more to supervised learning than regression.

In this section, we focus on classification problems where we put aside *how much?* questions and instead focus on *which category?* questions.

## Classification

Classification problems are often described in two slightly different ways:
- Hard Assignments: We are only interested in a single category (e.g. Is it spam? )
- Soft Assignments: We want to assess the *probability* that each cateogry applies.

This distinction is blured because often, even when we only care about hard assignments, we still use models that make soft assignments.

There are cases where more than one label might be true. 
This problem is commonly known as *multi-label classification*

--- 

To get started let's start with a simple image classification problem. 
Here, each input consists of a $2 \times 2$ image, giving us four featres $x_1, x_2, x_3, x_4$.
let's further assume that each image belongs to one category of three "cat", "chicken", "dog".

But how should we represent the labels. 
We have two obvious choices.
The most natural would be to choose $y \in \{1,2,3\}$, where each integer represents a category.
This is a great way if the categories had some natural order among them, This will make sense to cast it as a [order regression](https://en.wikipedia.org/wiki/Ordinal_regression) problem.

The other way is called *one-hot encoding*.
A one-hot encoding is a vector with all the components are set to 0 except one.

### Linear Model
To achieve a model that outputs *conditional probabilities* for each possible output, we must construct a model with multiple outputs.

To address multi-class classification using a linear model approach, we introduce as many *affine functions* as there are outputs. 

>While, strictly speaking, we could define one fewer function since the probability of the final category must be exactly $1$ minus the sum of the probabilities of all other categories,
for the sake of symmetry and computational simplicity, we use a slightly redundant parametrization.

Each output score, often referred to as a logit or raw score $o_i$, is the result of its own distinct affine transformation.

The calculations for the three outputs are explicitly defined as:

$$\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + b_1,\\
o_2 &= x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + b_2,\\
o_3 &= x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + b_3.
\end{aligned}
$$

![](./imgs/softmax_reg_single_layer.png)

### The Softmax
But how could we make sure that the outputs $o_i$ sum up to 1 just like probabilities, and each of them are nonnegative.

A way to accomplish this is to use exponential function which ensures  nonnegativity.
We can then transform these values so that they add up to 1 by dividing each by their sum.

Putting these two pieces together gives us the *softmax* function:

$$
\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o}) \quad \textrm{where}\quad \hat{y}_i = \frac{\exp(o_i)}{\sum_j \exp(o_j)}.
$$

> Note that the largest coordinate of $\mathbf{o}$
corresponds to the most likely class according to $\hat{\mathbf{y}}$.
Moreover, because the softmax operation
preserves the ordering among its arguments,
we do not need to compute the softmax
to determine which class has been assigned the highest probability.

After Vectorizing every thing we get this 

$$
 \begin{aligned} \mathbf{O} &= \mathbf{X} \mathbf{W} + \mathbf{b}, \\ \hat{\mathbf{Y}} & = \mathrm{softmax}(\mathbf{O}). \end{aligned} 
$$

## Loss Function
### Log Likelihood
Since the softmax functions gives us the conditinal probabilities of each class,
so we can check how probable the actual classes are by multipling the individual conditional probabilities:


$$P(Y|X) = \prod_{i=1}^n P(y^{(i)}|x^{(i)})$$

This is called the *likelihood*, We want to maxmize this, meaning finding the parameters that gives the highest probability to the actual observed classes.

Instead of maximizing the product of terms, we take the *negative logarithm of the likelihood* 

$$-\log P(Y|X) = \sum_{i=1}^n -\log P(y^{(i)}|x^{(i)}) = \sum_{i=1}^n l(y^{(i)}, \hat{y}^{(i)})$$


where for any pair of label $y$ 
and model prediction $\hat{y}$
over classes, the loss function is

$$l(y,\hat{y}) = - \sum_{j=1}^q y_j \log \hat{y}_j$$

This is commonly called *cross-entropy loss*

### Loss Simplification and Gradient
To better understand the function of the Softmax function and the Cross-Entropy Loss,
it is instructive to combine their definitions algebraically. 
By substituting the definition of the Softmax ($\hat{\mathbf{y}}_j = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)}$) 
into the Cross-Entropy Loss ($l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{\mathbf{y}}_j$),
we obtain the loss expressed purely in terms of the raw output scores, or logits ($\mathbf{o}$):

$$\begin{aligned}
l(\mathbf{y}, \hat{\mathbf{y}}) &=  - \sum_{j=1}^q y_j \log \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\
&= \sum_{j=1}^q y_j \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j \\
&= \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j.
\end{aligned}$$

The final simplification occurs because $\mathbf{y}$ is a one-hot vector: $\sum_{j=1}^q y_j = 1$. 


When we consider the derivative of the loss with respect to any single logit $o_j$,
we arrive at a remarkably simple and elegant result:

$$\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j.$$

This outcome is fundamental to machine learning optimization. 
It reveals that the gradient for Softmax Regression is conceptually identical to the gradient found in standard linear regression

While this is mathematically reasoable, it is really risky computationally, because of numerical overflow and underflow in the exponentiation

Recall that softmax computes probabilities via $\hat{y}_j = \frac{exp(o_j)}{\sum_k exp(o_k)}$,
So for a very large positive $o_k$ then the exponent might be larger than what python datatypes limits.
And the same goes for very negative numbers.

A way around this problem is to subtract the max $\bar{o} = max({o_k})$ as follows:

$$
 \hat y_j = \frac{\exp o_j}{\sum_k \exp o_k} =
   \frac{\exp(o_j - \bar{o}) \exp \bar{o}}{\sum_k \exp (o_k - \bar{o}) \exp \bar{o}} =
   \frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})}.
$$

By compining softmax and cross-entropy, we escape the numerical stability issues. We have:

$$
   \log \hat{y}_j =
   \log \frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})} =
   o_j - \bar{o} - \log \sum_k \exp (o_k - \bar{o}).
$$

This avoids both overflow and underflow. And now we can pass the logits and compute everything inside the cross-entropy loss function. 