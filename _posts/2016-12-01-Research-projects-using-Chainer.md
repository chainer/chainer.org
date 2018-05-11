---
title: Research projects using Chainer
layout: post
categories: General
author: Shunta Saito
---

Recently we found some great research projects that are using Chainer for their algorithm implementations and experiments. We searched for such publicly available projects on arXiv and summarized them here as a table that lists papers along with their URL links: [Research projects using Chainer](https://github.com/pfnet/chainer/wiki/Research-projects-using-Chainer).

As a brief overview on the field in which Chainer is used, we see a relatively large number of Natural Language Processing (NLP) papers in the above list. For example, "Quasi-Recurrent Neural Networks (QRNNs)" was recently proposed by James Bradbury and Stephen Merity et al. This is a 16x faster model for modeling sequential data with better predictive accuracy than stacked LSTMs of the same hidden size. The experiments in this paper were implemented in Chainer. Another interesting paper using Chainer is “Dynamic Coattention Networks for Question Answering”. This paper beats the state-of-the-art method for question answering on the Stanford question answering dataset, and all models are implemented and trained with Chainer. This amazing work was achieved by Caiming Xiong and Victor Zhong, et al. And actually, these two papers are both from MetaMind lead by Richard Socher. We’re happy and appreciate very much that they are using Chainer for such cutting edge research and their contribution to Chainer itself by sending PRs. Of course, there are many other projects that are also very cool in the list, and we deeply thank all the users that are trying Chainer.

As for the conference of the submitted papers, [International Conference on Learning Representations (ICLR)](http://www.iclr.cc) is the dominant conference that has papers using Chainer. ICLR is one of the top conferences, targets representation learning and actually focuses on Deep Learning, so that many papers tend to propose new architectures and those sometimes need very flexible framework to be implemented. These really motivate us to keep Chainer flexible and enabled to accelerate try-and-error cycles in research.

If you find other papers or cool projects using Chainer partially or wholly, we would appreciate it if you could let us know about them so that we may add them to the list.
