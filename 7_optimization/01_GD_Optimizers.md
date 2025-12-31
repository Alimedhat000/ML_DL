# Gradient Descent Methods

## Vanilla Gradient Descent

The goal of gradient descent is usually to minimize the loss function
$f(\theta)$ by moving in the direction of steepest descent- the negative gradient.

A good algorithm finds the minimum fast and reliably well
(i.e. it doesn’t get stuck in local minima, saddle points, or plateau regions, but rather goes for the global minimum).

For each parameter $\theta$, we update it based on the gradient $\nabla_\theta f(\theta)$ and a learning rate $\eta$:

$$\Delta = -\eta \cdot \nabla_\theta f(\theta)$$
$$\theta_{t+1} = \theta_t + \Delta$$

Imagine we only have two parameters to optimize,
and they are represented by the x and y dimension in the graph.
The surface is the loss function.
We want to find the (x, y) combination that’s at the lowest point of the surface.

Think of this as walking in the dark with only a flashlight.
You can see the slope at your feet and take one step at a time toward lower ground.
However, because it only looks at the current slope,
it can get stuck in local minima or move very slowly through flat plateaus.

![](https://contributor.insightmediagroup.io/wp-content/uploads/2025/03/image-19.gif)

## Momentum

Momentum simulates physical inertia.
Instead of stopping or turning instantly when the gradient changes,
the optimizer remembers its previous direction.

We introduce a velocity term $v$ and a decay rate (momentum coefficient) $\gamma$, usually set to $0.9$.
$$v_t = \gamma v_{t-1} + \eta \nabla_\theta f(\theta_t)$$
$$\theta_{t+1} = \theta_t - v_t$$

![Momentum(magenta) vs GD(cyan)](https://contributor.insightmediagroup.io/wp-content/uploads/2025/03/image-21.gif)

So, in what ways is Momentum better than vanilla gradient descent? In this comparison on the left, you can see two advantages:

1. Momentum simply moves faster (because of all the momentum it accumulates)

2. Momentum has a shot at escaping local minima
   (because the momentum may propel it out of a local minimum).
   In a similar vein, as we shall see later, it will also power through plateau regions better.

## AdaGrad

Instead of keeping track of the sum of gradient like momentum,
the **Ada**ptive **Grad**ient algorithm, or AdaGrad for short,
keeps track of the sum of gradient squared and uses that to adapt the gradient in different directions.
Often the equations are expressed in tensors. For each dimension:

The Math:We maintain a running sum of the squares of all historical gradients, $s_t$:

$$s_t = s_{t-1} + (\nabla_\theta f(\theta_t))^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t + \epsilon}} \cdot \nabla_\theta f(\theta_t)$$

![AdaGrad](https://contributor.insightmediagroup.io/wp-content/uploads/2025/03/image-136-1024x488.png)

In ML optimization, some features are very sparse.
The average gradient for sparse features is usually small so such features get trained at a much slower rate.
One way to address this is to set different learning rates for each feature, but this gets messy fast.

AdaGrad addresses this using the idea: the more you have updated a feature
already, the less you will update it in the future, thus giving chance for
other features to catch up.

his property allows AdaGrad (and other similar gradient-squared-based methods like RMSProp and Adam)
to escape a saddle point much better.
AdaGrad will take a straight path,
whereas gradient descent (or relatedly, Momentum) takes the approach of “let me slide down the steep slope first and maybe worry about the slower direction later”.

![](https://contributor.insightmediagroup.io/wp-content/uploads/2025/03/image-22.gif)

## RMSProp

The weakness of AdaGrad is that $s_t$ grows indefinitely, eventually making the learning rate so small that the model stops learning.
RMSProp (Root Mean Square Propagation) fixes this by using an exponentially weighted moving average.
The Math:
We add a decay factor $\beta$ (usually $0.9$) so that the optimizer "forgets" very old gradients:

$$s_t = \beta s_{t-1} + (1 - \beta) (\nabla_\theta f(\theta_t))^2$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t + \epsilon}} \cdot \nabla_\theta f(\theta_t)
$$

By keeping $s_t$ at a manageable size, RMSProp remains fast and effective throughout the entire training process.

To see the effect of the decaying, in this head-to-head comparison, AdaGrad white) keeps up with RMSProp (green) initially,
as expected with the tuned learning rate and decay rate.
But the sums of gradient squared for AdaGrad accumulate so fast that they soon become humongous (demonstrated by the sizes of the squares in the animation).

They take a heavy toll and eventually AdaGrad practically stops moving.
RMSProp, on the other hand, has kept the squares under a manageable size the whole time, thanks to the decay rate. This makes RMSProp faster than AdaGrad.

![](https://contributor.insightmediagroup.io/wp-content/uploads/2025/03/image-23.gif)

## Adam

Adam is the "Gold Standard" of optimizers. It combines the first moment (the mean of gradients from Momentum) and the second moment (the uncentered variance from RMSProp).

The Math: Adam tracks both $m_t$ (momentum) and $v_t$ (scaling):

1. First Moment (Momentum): $m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_\theta f(\theta_t)$
   2.Second Moment (Scaling): $v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_\theta f(\theta_t))^2$

2. Update:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}} m_t$$

Standard Defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$.

Adam essentially uses momentum to speed up the search and adaptive scaling to ensure every parameter is updated at the right pace. It is remarkably robust and usually the best starting choice for any deep learning project.
