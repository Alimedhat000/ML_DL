# Norms

Some of the most useful operators in linear algebra are *norms*.
Informally, the norm of a vector tells us how *big* it is. 
For instance, the $\ell_2$ norm measures
the (Euclidean) length of a vector.
Here, we are employing a notion of *size* that concerns the magnitude of a vector's components
(not its dimensionality). 

A norm is a function $\| \cdot \|$ that maps a vector
to a scalar and satisfies the following three properties:

1. Given any vector $\mathbf{x}$, if we scale (all elements of) the vector 
   by a scalar $\alpha \in \mathbb{R}$, its norm scales accordingly:
   $$\|\alpha \mathbf{x}\| = |\alpha| \|\mathbf{x}\|.$$
2. For any vectors $\mathbf{x}$ and $\mathbf{y}$:
   norms satisfy the triangle inequality:
   $$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|.$$
3. The norm of a vector is nonnegative and it only vanishes if the vector is zero:
   $$\|\mathbf{x}\| > 0 \textrm{ for all } \mathbf{x} \neq 0.$$

Many functions are valid norms and different norms 
encode different notions of size. 
The Euclidean norm that we all learned in elementary school geometry
when calculating the hypotenuse of a right triangle
is the square root of the sum of squares of a vector's elements.
Formally, this is called [**the $\ell_2$ *norm***] and expressed as

**$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}.$$**

The method `norm` calculates the $\ell_2$ norm.

[**The $\ell_1$ norm**] is also common 
and the associated measure is called the Manhattan distance. 
By definition, the $\ell_1$ norm sums 
the absolute values of a vector's elements:

**$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$**

Compared to the $\ell_2$ norm, it is less sensitive to outliers.
To compute the $\ell_1$ norm, 
we compose the absolute value
with the sum operation.


Both the $\ell_2$ and $\ell_1$ norms are special cases
of the more general $\ell_p$ *norms*:

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$
