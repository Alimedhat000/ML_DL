# Batch Normalization

A popular and effective technique that consistently accelerates the
convergence of deep networks. This together with [_res blocks_](./res_nets.md)
made it possible for reaserches to routinely train 100 layers deep networks.
Another important benefit is added because of the increased generalization.

When training deep networks,
Each layer has inputs that follow a certain distribution,
which shifts during training due to: the random starting values of the network,
and the natural variation in the input data.

This shifting pattern affecting the inputs to the network's inner layers
is called _internal covariate shift_.

Batch normalization is achieved through a normalization step that fixes
the means and variances of each layer's inputs.
But to use this with stochastic optimization methods it would be impractical
to use the global mean and variance.
Thus it is restrained to each minibatch in the **training process**.

Let us use $B$ to denote a minibatch of size $m$. The mean and variance of
$B$ is

$$
\mu_{B} = \frac{1}{m} \sum_{i=1}^{m} x_i
$$

$$
\sigma_{B}^{2} = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{B})^{2}
$$

For a layer of the network with d-dimensional input $x$,
each dimension of its input is then normalized i.e _re-centered_

$$
\hat{x}_{i}^{(k)} = \frac{x_{i}^{(k)} - \mu_{B}^{(k)}}{\sigma_{B}^{(k)}}
$$

Then the output of the batch normalization is as follows:

$$
{\displaystyle y_{i}^{(k)}=\gamma ^{(k)}{\hat {x}}_{i}^{(k)}+\beta ^{(k)}}
$$

Where the parameters $\gamma$ and $\beta$ are parameters that need to be
learned as part of the model training.
The resulting minibatch has zero mean and unit variance.

One thing to note is that during inference stage i.e _predection_,
the normalization step is computed with population mean and variance
i.e the variance of the whole dataset.
Since the parameters are fixed here the batch normalization is
essentially applying a linear transform to the activation function.

Although the original motivation centered on reducing
"internal covariate shift,"
practical experience shows that BN provides three main benefits:

1. Preprocessing: It keeps the intermediate estimation problem well-controlled by standardizing inputs.

2. Numerical Stability/Acceleration: It constrains the magnitude of intermediate variables, allowing for more aggressive learning rates.

3. Regularization: Serendipitously, the noise introduced by using minibatch statistics acts as a form of regularization, complementing techniques like dropout.
