---
title: ONNX support by Chainer
layout: post
categories: General
---

## ONNX support by Chainer

Today, we announce ONNX-Chainer, an open source Python package to export Chainer models to the Open Neural Network Exchange (ONNX) format. Preferred Networks joined the ONNX partner workshop yesterday that was held in Facebook HQ in Menlo Park, and discussed future direction of ONNX.

## What is ONNX?

[ONNX](http://onnx.ai/) is an open neural network exchange format for interchangeable neural network models. It enables the exchange of models between different frameworks, e.g., Chainer, PyTorch, MXNet, Caffe2, CNTK, etc. which support ONNX import/export.

ONNX uses Protocol Buffer as its serializing format and defines many layers commonly used in neural networks as operators. ONNX can support most neural network operands. The full list is provided here: [Operator Schemas](https://github.com/onnx/onnx/blob/master/docs/Operators.md). To convert a model to another framework, it's important to take care what operator types are supported by them to import/export. For example, if you want to export a network which has convolution layers from Chainer to another framework, you need to make sure that ONNX-Chainer can export convolution layers (it can) to ONNX and that the destination framework can import convolution layers from ONNX.

ONNX import and export are different functionalities which may require a different implementations. ONNX-Chainer is a tool to "export" a Chainer model to ONNX. At this time, it does not support importing an ONNX model into a Chainer model. To exchange models, check that: 1) the destination framework supports ONNX import, 2) all the layers used in the network you want to exchange are supported by both ONNX-Chainer and the ONNX import function of the destination framework.

FYI, current status of ONNX support in common frameworks can be found here: [Importing and Exporting from frameworks](https://github.com/onnx/tutorials#importing-and-exporting-from-frameworks). Chainer supports exporting only. This table does not include TensorRT, but it will support ONNX too according to this news article: [NGC Expands Further, with NVIDIA TensorRT Inference Accelerator, ONNX Compatibility, Immediate Support for MXNet 1.0](https://nvidianews.nvidia.com/news/nvidia-gpu-cloud-now-available-to-hundreds-of-thousands-of-ai-researchers-using-nvidia-desktop-gpus).

This blog post explains how to export a model written in Chainer into ONNX by using chainer/onnx-chainer.

## Install ONNX-Chainer

ONNX-Chainer is a Python package that depends on Chainer and ONNX, which also are Python packages. You can install it via pip:

```bash
pip install onnx-chainer
```

This command installs ONNX-Chainer, which supports the newest release of ONNX (v1.0). If you get error messages related to the ONNX installation, please see this instruction: [ONNX Installation](https://github.com/onnx/onnx#installation).

## Quick Start

ONNX-Chainer currently supports 50 operators defined in ONNX. The list of supported operators is here: [Supported Functions](https://github.com/chainer/onnx-chainer#supported-functions). Chainer's Define-by-Run approach does not determine the network architecture until it computes a forward pass. Thus, to convert a model to ONNX, ONNX-Chainer needs to run a forward computation on a given model once with dummy data that has the expected shape and type for the network. Once a forward pass is performed on the given model, the network architecture is determined, and ONNX-Chainer can trace the computational graph and convert it into ONNX. To provide this interface for ONNX-Chainer to trace the graph, models must have a `__call__` method that describes a forward computation. Let's look at example code to convert a convolutional network written in Chainer to ONNX format and save it as a `convnet.onnx` file.

```python
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import onnx_chainer

class ConvNet(chainer.Chain):

   def __init__(self, n_class):
       super(ConvNet, self).__init__()
       with self.init_scope():
           self.conv1 = L.Convolution2D(None, 6, 5, 1)
           self.conv2 = L.Convolution2D(6, 16, 5, 1)
           self.conv3 = L.Convolution2D(16, 120, 4, 1)
           self.fc4 = L.Linear(None, 84)
           self.fc5 = L.Linear(84, n_class)

   def __call__(self, x):
       h = F.max_pooling_2d(F.relu(self.conv1(x)), 2, 2)
       h = F.max_pooling_2d(F.relu(self.conv2(h)), 2, 2)
       h = F.relu(self.conv3(h))
       h = F.relu(self.fc4(h))
       return self.fc5(h)

model = ConvNet(n_class=10)

# ... train the model ...

# Prepare dummy data
x = np.zeros((1, 1, 28, 28), dtype=np.float32)

# Put Chainer into inference mode
chainer.config.train = False

# Convert the model to ONNX format
onnx_model = onnx_chainer.export(model, x, filename='convnet.onnx')
```

The `model` is a `chainer.Chain` object and `x` is dummy data that has the expected shape and type as the input to the model. To convert the model to ONNX format and save it as an ONNX binary, you can use the `onnx_chainer.export()` function. This function runs the given model once by giving the second argument directly to the model's `()` accessor. So you can give multiple arguments to the model by giving a list or dict to the second argument of the export function.

This function can also take some options. The details are following:

```
def export(model, args, filename=None, export_params=True, graph_name='Graph', save_text=False):

   Export function for chainer.Chain in ONNX format.

   This function performs a forward computation of the given Chain, model, by passing the given arguments args directly. It means, the output Variable object `y` to make the computational graph will be created by:

       y = model(*args)

   Args:
       model (~chainer.Chain): The model object you want to export in ONNX format. It should have __call__ method because the second argument args is directly given to the model by the () accessor.
       args (list or dict): The arguments which are given to the model directly.
       filename (str or file-like object): The filename used for saving the resulting ONNX model. If None, nothing is saved to the disk.
       export_params (bool): If True, this function exports all the parameters included in the given model at the same time. If False, the exported ONNX model does not include any parameter values.
       graph_name (str): A string to be used for the "name" field of the graph in the exported ONNX model.
       save_text (bool): If True, the text format of the output ONNX model is also saved with ".txt" extension.
   Returns:
       An ONNX model object.
```

## What's next?

You can find more detailed tutorials here: [ONNX tutorials](https://github.com/onnx/tutorials). We will add some ONNX-Chainer tutorials such as how to run a Chainer model with Caffe2 via ONNX, etc. to that repository.

We are also working on adding more supported operations of ONNX and are considering implementing importing functionality.

Microsoft blog: https://www.microsoft.com/en-us/cognitive-toolkit/blog/2018/01/open-ai-ecosystem-advances-preferred-network-adds-chainer-support-onnx-ai-format/