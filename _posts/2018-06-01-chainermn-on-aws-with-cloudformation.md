---
title: ChainerMN on AWS with CloudFormation
layout: post
categories: General
author: Shingo Omura
---

_Japanese version is [here](https://research.preferred.jp/2018/06/chainermn-on-aws-with-cloudformation/)_

[AWS CloudFormation][CFN] a service which helps us to practice [_Infrastructure As Code_](https://en.wikipedia.org/wiki/Infrastructure_as_Code) on wide varieties of AWS resources.  [AWS CloudFormation][CFN] provisions AWS resources in a repeatable manner and allows us to build and re-build infrastructure without time-consuming manual actions or write custom scripts.

Building distributed deep learning infrastructure requires some extra hustle such as installing and configuring deep learning libraries, setup ec2 instances, and optimization for computational/network performance.  Particularly, running [ChainerMN][ChainerMN] requires you to setup an MPI cluster.  [AWS CloudFormation][CFN] helps us automating this process.

Today, We announce [Chainer/ChainerMN pre-installed AMI][ChainerAMI] and [CloudFormaiton template for ChainerMN Cluster][ChainerCFN].  

- [chainer/chainer-ami][ChainerAMI]
- [chainer/chainer-cfn][ChainerCFN]

This enables us to spin up a [ChainerMN][ChainerMN] cluster on AWS and run your [ChainerMN][ChainerMN] tasks instantly in the cluster.

This article explains how to use them and how you can run distributed deep learning with [ChainerMN][ChainerMN] on AWS.

[Chainer AMI][ChainerAMI]
-----
The [Chainer AMI][ChainerAMI] comes with [Chainer][Chainer]/[CuPy][CuPy]/[ChainerMN][ChainerMN], its families ([ChianerCV][ChainerCV] and [ChainerRL][ChainerRL]) and [CUDA][CUDA]-aware [OpenMPI][OpenMPI] libraries so that you can run [Chainer][Chainer]/[ChainerMN][ChainerMN] workloads easily on AWS EC2 instances even on ones with GPUs.  This image is based on [AWS Deep Learning Base AMI](https://docs.aws.amazon.com/dlami/latest/devguide/overview-base.html).

The latest version is `0.1.0`.  The version includes:

- OpenMPI version `2.1.3`
  - it was built only for `cuda-9.0`.
- All Chainer Families (they are built and installed against both `python` and `python3` environment)
  - `CuPy` version `4.1.0`
  - `Chainer` version `4.1.0`,
  - `ChainerMN`, version `1.3.0`
  - `ChainerCV` version `0.9.0`
  - `ChainerRL` version `0.3.0`


[CloudFormation Template For ChainerMN][ChainerCFN]
---
This template automatically sets up a [ChainerMN][ChainerMN] cluster on AWS.  Here's the setup overview for AWS resources:

- VPC and Subnet for the cluster (you can configure existing VPC/Subnet)
- S3 Bucket for sharing ephemeral ssh-key, which is used to communicate among MPI processes in the cluster
- Placement group for optimizing network performance
- ChainerMN cluster which consists of:
  - `1` master EC2 instance
  - `N (>=0)` worker instances (via AutoScalingGroup)
  - `chainer` user to run mpi job in each instance
  - `hostfile` to run mpi job in each instance
- (Option) [Amazon Elastic Filesystem][EFS] (you can configure an existing filesystem)
  -  This is mounted on cluster instances automatically to share your code and data.
- Several required SecurityGroups, IAM Role

The latest version is `0.1.0`.  Please see [the latest template](https://s3-us-west-2.amazonaws.com/chainer-cfn/chainer-cfn-v0.1.0.template) for detailed resource definitions.

As stated on our [recent blog on ChainerMN 1.3.0](https://chainer.org/general/2018/05/25/chainermn-v1-3.html),  using new features (double buffering and all-reduce in half-precision floats) enables almost linear scalability on AWS even at ethernet speeds.

How to build a [ChainerMN][ChainerMN] Cluster with the [CloudFormation Template][ChainerCFN]
---
This section explains how to setup [ChainerMN][ChainermN] cluster on AWS in a step-by-step manner.

First, please click the link below to create [AWS CloudFormation][CFN] Stack. And just click 'Next' on the page.

[![launch stack](https://s3.amazonaws.com/cloudformation-examples/cloudformation-launch-stack.png)](https://console.aws.amazon.com/cloudformation/home#/stacks/new?stackName=chainermn-sample&templateURL=https://s3-us-west-2.amazonaws.com/chainer-cfn/chainer-cfn-v0.1.0.template)

In "Specify Details" page, you can configure parameters on stack name, VPC/Subnet, Cluster, EFS configurations.  The screenshot below is an example for configuring `4` `p3.16xlarge` instances, each of which has 8 NVIDIA Tesla V100 GPUs.

![chainer-cfn-specifying-details]({{ site.baseurl }}/images/chainer-cfn-specifying-details.png)

At the last confirmation page, you will need to check a box in CAPABILITY section because this template will create some IAM roles for cluster instances.

![chainer-cfn-specifying-details]({{ site.baseurl }}/images/chainer-cfn-capabilities-confirmation.png)

After several minutes (depending on cluster size), the status of the stack should converge to `CREATE_COMPLETE` if all went well, meaning your cluster is ready. You can access the cluster with `ClusterMasterPublicDNS` which will appear in the output section of the stack.

How to run [ChainerMN][ChainerMN] Job in the Cluster
--
You can access the cluster instances with keypair which was specified in template parameter.

```
ssh -i keypair.pem ubuntu@ec2-ww-xxx-yy-zzz.compute-1.amazonaws.com
```

Because [Chainer AMI][ChainerAMI] comes with all required libraries to run [Chainer][Chainer]/[ChainerMN][ChainerMN] jobs, you only need to download your code to the instances.

```
# switch user to chainer
ubuntu@ip-ww-xxx-yy-zzz$ sudo su chainer

# download ChainerMN's train_mnist.py into EFS
chainer@ip-ww-xxx-yy-zzz$ wget https://raw.githubusercontent.com/chainer/chainermn/v1.3.0/examples/mnist/train_mnist.py -O /efs/train_mnist.py
```

That's it!  Now, you can run MNIST example with [ChainerMN][ChainerMN] by just invoking `mpiexec` command.

```
# It will spawn 32 processes(-n option) among 4 instances (8 processes per instance (-N option))
chainer@ip-ww-xxx-yy-zzz$ mpiexec -n 32 -N 8 python /efs/train_mnist.py -g
...(you will see ssh warning here)
==========================================
Num process (COMM_WORLD): 32
Using GPUs
Using hierarchical communicator
Num unit: 1000
Num Minibatch-size: 100
Num epoch: 20
==========================================
epoch       main/loss   validation/main/loss  main/accuracy  validation/main/accuracy  elapsed_time
1           0.795527    0.316611              0.765263       0.907536                  4.47915
...
19          0.00540187  0.0658256             0.999474       0.979351                  14.7716
20          0.00463723  0.0668939             0.998889       0.978882                  15.2248

# NOTE: above output is actually the output of the second try because mnist dataset download is needed in the first try.
```

[CFN]: https://aws.amazon.com/cloudformation/
[EFS]: https://aws.amazon.com/efs/features/
[ChainerAMI]: https://github.com/chainer/chainer-ami
[ChainerCFN]: https://github.com/chainer/chainer-cfn
[ChainerMN]: https://github.com/chainer/chainermn
[Chainer]: https://chainer.org
[CuPy]: https://cupy.chainer.org/
[ChainerCV]: https://github.com/chainer/chainercv
[ChainerRL]: https://github.com/chainer/chainerrl
[CUDA]: https://developer.nvidia.com/cuda-zone
[OpenMPI]: https://www.open-mpi.org/
