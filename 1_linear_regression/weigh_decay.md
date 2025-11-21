# Weight Decay

After discussing the problem of overfitting, we can introduce our first *reqularization* technique.
Recall that we can always mitigate overfitting by collecting more data. 
However, that's too costly.
So for now, we can assume that we already have as much data as resources permit.

## What is Weight Decay
*Weight decay* operates by restricting the values that the parameters can take, simply penalizing large weights.
More commonly called $l2$ regularization outside of deep learining when optimized by minibatch stochastic gradient descent.

The technique is motivated by the basic intuition that amon all functions $f$,
The function $f = 0 $ is the *simplest*,
and that we can measure the complexity of a function by the distance of it's parameters from zero.
But how excatily can we measure this?

One simple interpration is to measure the complexity of a linear functio $f(x) = \mathbf{w^Tx}$ by some norm of its weigt vector, $||\mathbf{w}||^2$. 
Recall the $l2$ and $l1$ norms from [Norms](/linear_algebra/norms.md).

The most common method for ensuring a small weight vector is to add it's norm as a penalty term to the problem of minimizing the loss.

Now, if our weight vector grows too large, our learning algorithm might focus on minimizing the weight norm $||\mathbf{w}||^2$ rather than minimizing the training error. 

$$
\min \{L(\mathbf{w},b) + \frac{\lambda}{2}||\mathbf{w}||^2\} 
$$

In practice we characterize this trade-off via the *regularization* constant $\lambda$,a nonnegative hyperparameter that we fit using validation data.

For $\lambda = 0$, we use our original loss function. For $\lambda \lt 0$, we restrict the size of $||\mathbf{w}||$.

Using the $l2$ norm is not the only valid choice. in fact, other choices are also popular throughout statistics.
While $l2$-regularized linear models constitute the classic *ridge regression* algorithm, $l1$-regularized linear regression is similarly a fundemental method called *lasso regression*.

The difference between them is that $l2$ norm places an outsize penalty on large components of the weight vector.
This leads towards models the distribute evenly across the features.

By contrast $l1$ norm penalties lead to models that concentrate weights on small set of features by clearing others to zero.
This gives us an effective method for *feature selection*, which may be desirable for other reasons. 
For example, if our model only relies on a few features, then we may not need to collect, store, or transmit data for the other (dropped) features.