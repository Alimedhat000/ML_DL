# Forward and Backward Propagation

So far, when we trained our models we only implemented the calculations of the forward propagation,
but when it came to calculate the gradients we just invoked the `backward` method.

## Forward Propagation

_Forward propagation_ or _forward pass_ refers to the calculation and storage of intermediate variables
for a neural network in order from input to output.

Assuming an input $\mathbf{x} \in \mathbb{R}^d$ and a network with one hidden layer
with no bias for simplicity,
calculations proceeds as follows:

1. **Hidden Layer Intermediate Variable (z)**:

   The input is transformed by the weight matrix $W_1$ to produce the intermediate variable:

$$
z = W_1 \mathbf{x}
$$

2. **Hidden Activation Vector (h)**:
   The intermediate variable $z$ is passed through the activation function $\sigma$ to produce the hidden activation vector $h$:

$$
h = \sigma(z)
$$

3. **Output Layer Intermediate Variable (y)**:

The hidden activation vector $h$ is transformed by the weight matrix $W_2$ to produce the output layer intermediate variable $y$:

$$
y = W_2 h
$$

4. **Loss Term (L)**:

   The output layer intermediate variable $y$ is passed through a loss function $L$ to produce the loss term $L$:

$$
L = L(y, \mathbf{y})
$$

5. **Regularization Term (s)**:

This would be discussed later but here's the definenition:

$$
\mathbf{s} = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_F^2 + \|\mathbf{W}^{(2)}\|_F^2\right)
$$

Where the Frobenius norm of the matrix is the $l2$ norm applied after
flattening the matrix into a vector.

6. Regularized Loss ($J$):

Finally, the regularized loss is given by:

$$
J = L + s
$$

## Back Propagation

_back propagation_ is the method that calculates the gradients of the neural network.
In short, the method traverses the network in reverse order, from the output to the input,
according to the chain rule.

The objective of back propagation is to calculate the gradients
$\frac{\partial J}{\partial W^{(1)}}$ and $\frac{\partial J}{\partial W^{(2)}}$.
To accomplish this, we start from the regularized loss and make our way backward to the
parameters.

The gradients of the regularized loss $J = L +s$ with respect to loss and s:

$$
\frac{\partial J}{\partial L} = \frac{\partial J}{\partial s} = 1
$$

We get the $\frac{\partial J}{\partial W^{(2)}}$ from:

1. From the loss $L$ we get from $y$ to $W^{(2)}$.
   Since $y = W^{(2)}h$,
   the derivative of the output w.r.t the weights is the input to that layer ($\mathbf{h}$).

In matrix calculus,
to make the dimensions match,
we multiply the error vector
$\frac{\partial J}{\partial \mathbf{o}}$ by the transpose of the input vector $\mathbf{h}^\top$.

2. From Regularization: The derivative of $\frac{\lambda}{2} \|\mathbf{W}^{(2)}\|^2$ is simply $\lambda \mathbf{W}^{(2)}$.

We get the $\frac{\partial J}{\partial \mathbf{W}^{(1)}}$:

$$\frac{\partial J}{\partial \mathbf{W}^{(1)}} = \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}$$

---
