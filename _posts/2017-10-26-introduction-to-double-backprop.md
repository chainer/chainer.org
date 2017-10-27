---
title: Introduction to double backprop
layout: post
categories: General
---

# Introduction

Starting from Chainer v3, we can define a model that involves higher order derivatives and optimize it with gradient-based optimization algorithms. In some DL frameworks, this feature is called double backpropagation (a.k.a. differentiable gradient, grad of grad). This article is a brief introduction to double backpropagation and how we implement it in Chainer. This article is based on the presentation in Chainer Meetup #6 @Preferred Networks, Tokyo, Japan on Sep. 30th 2017. The slides are available from [here](https://www.slideshare.net/KentaOono/comparison-of-deep-learning-frameworks-from-viewpoint-of-double-backpropagation).

# Why do we need double backpropagation?

Recently there have been several models whose optimization is reduced to a minimization or maximization of the function of the form above. Wasserstein GAN (WGAN)[1] is a variant of Generative Adversarial Networks (GAN), which have been attracted attention because of the clear and natural images that the model generates. Soon after the model was proposed, Gulrajani et al. proposed in [2] an improved optimization technique called WGAN-GP (gradient penality) that trains the model so that the gradient of the output of loss value should be close to 1. The loss function of this model is as follows:

$$
\mathbb{E}_{x\sim \mathbb{P}_g} [D(x)] - \mathbb{E}_{x\sim \mathbb{P}_r} [D(x)] + \lambda \mathbb{E}_{x\sim \mathbb{P}_x} [(\| \nabla_{x} D(x) \|_2 - 1)^2 ]
$$

Also, some reinforcement learning algorithms (e.g. Trust Region Policy Optimization (TRPO) [3]) or meta-learning algorithms (e.g. [4][5]) also need second derivatives.

# Neural networks as a computational graph

In most DL frameworks, a neural network is conceptualized as a computational graph which is a bipartite directed acyclic graph (DAG) consisting of variable and function nodes. The computational graph is used to keep track of the history of computation for backward propagation. For example, let `x1`, `x2` be variables and compute `y` and `z` as `y = x1 * x2` and `z = y + x3`, respectively. These are represented as follows:

![neural network as a computational graph]({{ site.url }}/assets/neural_network_as_a_computational_graph.png)

# Two types of backpropagation

Roughly speaking, there are two ways to execute backpropagation on computational graphs. One is backpropagation through graph in which we inductively from the downstream of computational graph. The other is to build a computational graph for backpropagation. Once a graph for backpropagation is built, we should compute the graph forward.

![two types of backpropagation]({{ site.url }}/assets/two_types_of_backpropagation.png)

While implementation of the former approach is easier, some models are difficult to implement. The most important ones is a model that involves the gradient of some function, that is, a function $L$ of the form

$$L(x) = G(f(x), ∇f(x)))$$

for some differentiable function $f:\mathbb{R}^d \to \mathbb{R}$ and $g: \mathbb{R} \times \mathbb{R}^d\to \mathbb{R}$. Here, $\nabla f$ is a gradient of $f$. Users have to compute the gradient of the function and build the corresponding graph manually to optimize the model with gradient-based approach. This is because optimization involves the gradient of the gradient, that is, second derivative.

On the other hand, in the latter approach, we can compute the gradient of some variable by just calling backpropagation, as gradient is represented as a variable in a computational graph, we can optimize the model in the same way as models without gradients.

# How Chainer implements double backprop

Previously, Chainer took the former approach to implement backpropagation. Therefore, we could not higher order derivatives automatically. Therefore, we move the design principle to the latter approach. Specifically, we introduced FunctionNode that represents an operation node in computational graphs. The most important difference between FunctionNode and Function is that its backward method is implemented in Variable level, not NumPy level. Let’s consider the function that computes $F(x, y) = x^2 + y$ where $x, y \in \mathbb{R}^d$ and multiplication is done in a element-wise manner. The pseudo code is as follows:

```python
# The API of FunctionNode is changed for simplicity (see note below).
class F(FunctionNode):
    def forward(self, inputs):
        x, y = inputs
        return x * x + y
    def backward(self, inputs, grad_outputs):
        x, _ = inputs
        gz, = grad_outputs
        return 2 * gz * x, gz
```

The variables x and y in forward method is either NumPy (for CPU) or CuPy (for GPU), whereas x and gz in backward are instances of chainer.Variable. So, by calling backward method, Chainer constructs a computational graph for backward, which enable us further backpropagation.

### Note
The API of the backward method in FunctionNode is different from actual one. Specifically, the second argument specifies indexes that we want to backpropagate. The backward method should compute gradients with respect to the inputs whose indexes are in the second argument.
Some cautious reader could notice that, how we can get input values (x in this case) that are necessary to calculate gradients, although inputs are not available as arguments? Chainer provides `FunctionNode.retain_inputs` to retain inputs for backward computation and `self.get_retained_inputs` method to retrieve the values. The actual implementation is as follows:

```python
class F(FunctionNode):
    def forward(self, inputs):
        x, y = inputs
        self.retain_inputs((0,))
        return x * x + y
    def backward(self, indexes, grad_outputs):
        x, = self.get_retained_inputs()
        z, = grad_outputs
        ret = []
        if 0 in indexes:
            ret.append(2 * gz * x)
        if 1 in indexes:
            ret.append(gz)
        return ret
```


We can use Function object to implement differentiable functions. But it does not support double backpropagation.

# Double backpropagation

From now on, we assume that `F` is a scalar-valued function (i.e. $F(x, y) \in \mathbb{R}$).

In the picture below, we compute a scalar loss value `L` from `z = F(x, y)` in differentiable manner.
By invoking backpropagation from `L`, we can compute the derivative of L with respect to x and y respectively by extending the computational graph:

![how to double backprop 1]({{ site.url }}/assets/double_backprop_1.png)

As a special case, if `L` is `z` itself, we can create a node that represents derivatives of `z` with respect to `x` and `y`. We call them `gx` and `gy`, respectively. Note that `gx` (resp. `gy`) have a same shape as `x` (resp. `y`).

![how to double backprop 2]({{ site.url }}/assets/double_backprop_2.png)

Suppose `x` is (therefore `gx` is) scalar.
If we invoke backpropagation from `gx `, we can compute the second derivative of `z` with respect to `x`.

![how to double backprop 3]({{ site.url }}/assets/double_backprop_3.png)

Usually the backward method of a variable computes gradients with respect to all variables that  the variable depends on. But sometimes we only need a part of them. To reduce computational const, Chainer (and also other deep learning frameworks that support double backpropagation) offers `grad` method that calculates gradients of output variable with respect to inputs specified only. In the example above, we can calculate gx as:

```python
gx, = chainer.grad([z], [x], enable_double_backprop=...)
```

Here, `enable_double_backprop` should be `True` if we want to backpropagate further from the computed gradients.


### Note
Someone could think that we can compute Hessian with this feature. But what we actually compute is a *Hessian-vector product*. Specifically, if $x$ a $d$-dimensional vector and if we set $g$, which has same dimensionality as $x$, as a gradient of $gx$ and invoke backpropagation. The point is that we accumulate the gradient when invoking backward propagation. $ggx$ is a $d$-dimensional vector whose $i$-th element is

$$
ggx_i = \sum_{j=1}^d g_j \frac{\partial}{\partial x_i} gx_j.
$$ 

Since $gx$ is a derivative of $z$ w.r.t. $x$, it is equal to

$$ 
\sum_{j=1}^{d} g_j \frac{\partial^2 z}{\partial x_j \partial x_i} = Hg,
$$

where $H$ is a Hessian matrix of z with respect to x. It is nothing but a Hessian-vector product of the given gradient $g$.


# Optimization of functions that involve derivatives

Now let’s go back to the functions of the form $L(x) = G(f(x), ∇f(x))$ where $f:\mathbb{R}^d \to \mathbb{R}$ and $g: \mathbb{R} \times \mathbb{R}^d\to \mathbb{R}$. We have explained tools that is needed to optimize this function.

![how to double backprop 4]({{ site.url }}/assets/double_backprop_4.png)


1. We compute `z = f(x)` with constructing a computational graph.
2. Use grad method to extend the computational graph to get `gx` (or call `backward` method of `z` also works).
3. Construct further the part of the computational graph that corresponds to `G` to get a final value `L`.
4. Call backward of `L` to get the derivative of `L` with respect to x.

The following code is a toy example of manual optimization of function that have gradients of functions. In this example we optimize $f(x) = (x - 1)^2$ where x in R. But we artificially introduce the gradient into the definition of f by taking the gradient of $x^3 / 3$ to compute the square of $x$.

```python
import chainer
from chainer import Variable
import numpy as np

def f(x):
    y = x * x * x / 3
    gx, = chainer.grad([y], [x], enable_double_backprop=True)
    z = gx - 2 * x + 1  # z = (x - 1) ** 2
    return z

x = Variable(np.random.uniform(size=()))
for _ in range(30):
    x = Variable(x.data)
    z = f(x)
    z.backward()
    x -= 0.1 * x.grad
    print(x.data)
```

<!-- Also several people have already implemented WGAN with the RC version of Chainer v3. Code is available is from here. -->

# Conclusion

In this article, we introduced the basics of double backpropagation and how we implement it in Chainer. There are two ways how we back propagate with computational graphs: backward through graphs and backward by extending graphs. To support double backpropagation, Chainer have changed its internal design and the APIs of function nodes to take the latter approach. With this change, we can implement models that need gradients in the forward propagation. Such a type of models is increasingly popular in various fields in deep learning including generative models, reinforcement learning, or meta-learning.

Currently, we are working on making existing Chainer functions support double backpropagation. We welcome any kind of contribution from you all e.g. wider support of double backpropagationcoding examples that use double backpropagation and so on.


# References

[1] Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein gan. arXiv preprint arXiv:1701.07875.

[2] Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017). Improved training of wasserstein gans. arXiv preprint arXiv:1704.00028.

[3] Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). Trust region policy optimization. In Proceedings of the 32nd International Conference on Machine Learning (ICML-15) (pp. 1889-1897).

[4] Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. arXiv preprint arXiv:1703.03400.

[5] Andrychowicz, M., Denil, M., Gomez, S., Hoffman, M. W., Pfau, D., Schaul, T., & de Freitas, N. (2016). Learning to learn by gradient descent by gradient descent. In Advances in Neural Information Processing Systems (pp. 3981-3989).

