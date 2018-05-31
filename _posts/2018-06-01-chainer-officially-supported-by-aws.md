---
title: Open source deep learning framework Chainer officially supported by Amazon Web Services
layout: post
categories: General
---

Chainer has worked with Amazon Web Services (AWS) to provide access to the Chainer deep learning framework as a listed choice across many of AWS applications. Chainer provides straightforward calculation of deep neural networks in Python. The combination with AWS leverages Chainer’s exceptional abilities in multi-GPU and multi-server scaling, as demonstrated when [PFN trained ResNet50 on ImageNet-1K using Chainer in 15 minutes](https://www.preferred-networks.jp/docs/imagenet_in_15min.pdf), four times faster than the previous record held by Facebook.

Usage of multi-GPU and multi-server scaling allows researchers to leverage the ability of the cloud to provide computing resources on demand. Chainer’s unparalleled ability for parallel computing combined with AWS cloud resources available on demand enables researchers and engineers to minimize their cost while training complex deep learning models in a fraction of the time required on more limited hardware.

Chainer is already available as part of the AWS Deep Learning Amazon Machine Image (AMI). This is further enhanced by Chainer’s recent release of a CloudFormation script, which enables easy deployment of multiple Chainer AMIs at a time. Chainer has been tested to provide 95% scaling efficiency up to 32 GPUs on AWS, which means training of a neural network can be done up to thirty times as fast.

To simplify the process of pre-processing data, tuning hyperparameters, and deploying a neural network, Chainer is now supported on Amazon SageMaker. Amazon SageMaker is a fully-managed platform that enables developers and data scientists to quickly and easily build, train, and deploy machine learning models at any scale. Using Chainer on Sagemaker will provide speed increases from parallelization, in addition to the deployment benefits of SageMaker.

In an additional announcement, AWS now supports Chainer on AWS Greengrass, the AWS service that lets you run local compute, messaging, data caching, sync, and ML inference capabilities for connected devices in a secure way. Combined with Amazon SageMaker, this allows access to the ease and speed of Chainer when training models on SageMaker and direct deployment on AWS Greengrass to IoT devices.

The Chainer team is excited about these releases by AWS and looks forward to providing further advances as deep learning techniques continue to advance.
