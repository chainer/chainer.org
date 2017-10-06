---
title: How to use Chainer for Theano users
layout: post
categories: General
---

As we [mentioned on our blog](https://chainer.org/general/2017/09/29/thank-you-theano.html), Theano will stop development in a few weeks. Many aspects of Chainer were inspired by Theano's clean interface design, so we would like to introduce Chainer to users of Theano. We hope this article assists interested Theano users to move to Chainer easily.

First, let's summarize the key similarities and differences between Theano and Chainer.

### Key similarities:

- Python-based library
- Functions can accept NumPy arrays
- CPU/GPU support
- Easy to write various operation as a differentiable function (custom layer)

### Key differences:

- Theano compiles the computational graph before run
- Chainer builds the comptuational graph in runtime
- Chainer provides many high-level APIs for neural networks
- Chainer supports distributed learning with ChainerMN


In this post, we asume that the modules below have been imported.


```python
import numpy as np
```


```python
import theano
import theano.tensor as T
```


```python
import chainer
import chainer.functions as F
import chainer.links as L
```


## Define a parametric function

A neural network basically has many parametric functions and activation functions, commonly called "layers." Let's see the difference between how to create a new parametric function between Theano and Chainer. In this example, to show the way to do the same thing with the two different libraries, we show how to define the 2D convolution function. Chainer has `chainer.links.Convolution2D`, so it isn't necessary to write the code below to use 2D convolution as a building block of a network.

### Theano:


```python
class TheanoConvolutionLayer(object):

    def __init__(self, input, filter_shape, image_shape):
        # Prepare initial values of the parameter W
        spatial_dim = np.prod(filter_shape[2:])
        fan_in = filter_shape[1] * spatial_dim
        fan_out = filter_shape[0] * spatial_dim
        scale = np.sqrt(3. / fan_in)

        # Create the parameter W
        W_init = np.random.uniform(-scale, scale, filter_shape)
        self.W = theano.shared(W_init.astype(np.float32), borrow=True)

        # Create the paramter b
        b_init = np.zeros((filter_shape[0],))
        self.b = theano.shared(b_init.astype(np.float32), borrow=True)

        # Describe the convolution operation
        conv_out = T.nnet.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape)

        # Add a bias
        self.output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        # Store paramters
        self.params = [self.W, self.b]
```

How can we use this class? In Theano, the computation is defined as code using symbols, but doesn't perform actual computation at that time. Namely, it defines the computational graph before run. To use the defined computational graph, we need to define another operator using `theano.function` which takes input variables and output variables.


```python
batchsize = 32
input_shape = (batchsize, 1, 28, 28)
filter_shape = (6, 1, 5, 5)

# Create a tensor that represents a minibatch
x = T.fmatrix('x')
input = x.reshape(input_shape)

conv = TheanoConvolutionLayer(input, filter_shape, input_shape)
f = theano.function([input], conv.output)
```

`conv` is the definition of how to compute the output from the first argument `input`, and `f` is the actual operator. You can pass values to `f` to compute the result of convolution like this:


```python
x_data = np.random.rand(32, 1, 28, 28).astype(np.float32)

y = f(x_data)

print(y.shape, type(y))
```

    (32, 6, 24, 24) <class 'numpy.ndarray'>


### Chainer:

What about the case in Chainer? Theano is a more general framework for scientific calculation, while Chainer focuses on neural networks. Chainer has many high-level APIs to write the building blocks of neural networks easier. Well, how to write the same convolution operator in Chainer?


```python
class ChainerConvolutionLayer(chainer.Link):

    def __init__(self, filter_shape):
        super().__init__()
        with self.init_scope():
            # Specify the way of initialize
            W_init = chainer.initializers.LeCunUniform()
            b_init = chainer.initializers.Zero()

            # Create a parameter object
            self.W = chainer.Parameter(W_init, filter_shape)          
            self.b = chainer.Parameter(b_init, filter_shape[0])

    def __call__(self, x):
        return F.convolution_2d(x, self.W, self.b)
```

Actually, Chainer has pre-implemented `chainer.links.Convolution2D` class for convolution. So, you don't need to implement the code above by yourself, but it shows how to do the same thing written in Theano above.

You can create your own parametric function by defining a class inherited from `chainer.Link` as shown in the above. What computation will be applied to the input is described in `__call__` method.

Then, how to use this class?


```python
chainer_conv = ChainerConvolutionLayer(filter_shape)

y = chainer_conv(x_data)

print(y.shape, type(y), type(y.array))
```

    (32, 6, 24, 24) <class 'chainer.variable.Variable'> <class 'numpy.ndarray'>


Chainer provides many functions in `chainer.functions` and it takes NumPy array or `chainer.Variable` object as inputs. You can write arbitrary layer using those functions to make it differentiable. Note that a `chainer.Variable` object contains its actual data in `array` property.

**NOTE:**
You can write the same thing using `L.Convolution2D` like this:


```python
conv_link = L.Convolution2D(in_channels=1, out_channels=6, ksize=(5, 5))

y = conv_link(x_data)

print(y.shape, type(y), type(y.array))
```

    (32, 6, 24, 24) <class 'chainer.variable.Variable'> <class 'numpy.ndarray'>


## Use Theano function as a layer in Chainer

How to port parametric functions written in Theano to `Link`s in Chainer is shown in the above chapter, but there's an easier way to port **non-parametric functions** from Theano to Chainer.

Chainer provides [`TheanoFunction`](https://docs.chainer.org/en/latest/reference/generated/chainer.links.TheanoFunction.html?highlight=Theano) to wrap a Theano function as a `chainer.Link`. What you need to prepare is just the inputs and outputs of the Theano function you want to port to Chainer's `Link`. For example, a convolution function of Theano can be converted to a Chainer's `Link` as followings:


```python
x = T.fmatrix().reshape((32, 1, 28, 28))
W = T.fmatrix().reshape((6, 1, 5, 5))
b = T.fvector().reshape((6,))
conv_out = T.nnet.conv2d(x, W) + b.dimshuffle('x', 0, 'x', 'x')

f = L.TheanoFunction(inputs=[x, W, b], outputs=[conv_out])
```


It converts the Theano computational graph into Chainer's computational graph! So it's differentiable with the Chainer APIs, and easy to use as a building block of a network written in Chainer. But it takes `W` and `b` as input arguments, so it should be noted that it doesn't keep those parameters inside.

Anyway, how to use this ported Theano function in a network in Chainer?


```python
class MyNetworkWithTheanoConvolution(chainer.Chain):

    def __init__(self, theano_conv):
        super().__init__()
        self.theano_conv = theano_conv
        W_init = chainer.initializers.LeCunUniform()
        b_init = chainer.initializers.Zero()
        with self.init_scope():
            self.W = chainer.Parameter(W_init, (6, 1, 5, 5))
            self.b = chainer.Parameter(b_init, (6,))
            self.l1 = L.Linear(None, 100)
            self.l2 = L.Linear(100, 10)

    def __call__(self, x):
        h = self.theano_conv(x, self.W, self.b)
        h = F.relu(h)
        h = self.l1(h)
        h = F.relu(h)
        return self.l2(h)
```

This class is a Chainer's model class which is inherited from `chainer.Chain`. This is a standard way to define a class in Chainer, but, look! it uses a Theano function as a layer inside `__call__` method. The first layer of this network is a convolution layer, and that layer is Theano function which runs computation with Theano.

The usage of this network is completely same as the normal Chainer's models:


```python
# Instantiate a model object
model = MyNetworkWithTheanoConvolution(f)

# And give an array/Variable to get the network output
y = model(x_data)
```

    /home/shunta/.pyenv/versions/anaconda3-4.4.0/lib/python3.6/site-packages/chainer/utils/experimental.py:104: FutureWarning: chainer.functions.TheanoFunction is experimental. The interface can change in the future.
      FutureWarning)


This network takes a mini-batch of images whose shape is `(32, 1, 28, 28)` and outputs 10-dimensional vectors for each input image, so the shape of the output variable will be `(32, 10)`:


```python
print(y.shape)
```

    (32, 10)


This network is differentiable and the parameters of the Theano's convolution function which are defined in the constructer as `self.W` and `self.b` can be optimized through Chainer's optimizers normaly.


```python
t = np.random.randint(0, 10, size=(32,)).astype(np.int32)
loss = F.softmax_cross_entropy(y, t)

model.cleargrads()
loss.backward()
```

You can check the gradients calculated for the parameters `W` and `b` used in the Theano function `theano_conv`:


```python
W_gradient = model.W.grad_var.array
b_gradient = model.b.grad_var.array
```


```python
print(W_gradient.shape, type(W_gradient))
print(b_gradient.shape, type(b_gradient))
```

    (6, 1, 5, 5) <class 'numpy.ndarray'>
    (6,) <class 'numpy.ndarray'>


While we are familiar with Chainer, it has been longer since we have used Theano. If there are corrections or additional advice we should add to this guide, please let us know.

Contact:

**Forum** ([en](https://groups.google.com/forum/#!forum/chainer), [ja](https://groups.google.com/forum/#!forum/chainer-jp))
| **Slack invitation** ([en](https://bit.ly/chainer-slack), [ja](https://bit.ly/chainer-jp-slack))
