---
title: ChainerMN on Kubernetes with GPUs
layout: post
categories: General
---

[Kubernetes](https://kubernetes.io/) is today the most popular open-source system for automating deployment, scaling, and management of containerized applications.  As the rise of [Kubernetes](https://kubernetes.io/), bunch of companies are running [Kubernetes](https://kubernetes.io/) as a platform for various workloads including web applications, databases, cronjobs and so on.  Machine Learning workloads, including Deep Learning workloads, are not an exception even though such workloads require sepcial hardwares like GPUs.

[Kubernetes](https://kubernetes.io/) can [schedule NVIDIA GPUs by default](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/).  So, single node [Chainer](https://chainer.org/) workloads are straightforward.  You can simply launch a `Pod` or a `Job` with [`nvidia.com/gpu` resource request](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/).

However running [ChainerMN](https://github.com/chainer/chainermn/) on [Kubernetes](https://kubernetes.io/) is not straightforward because it requires us to setup an MPI cluster. [Kubeflow](https://github.com/kubeflow/kubeflow) can be a big help for it. The [Kubeflow](https://github.com/kubeflow/kubeflow) project is dedicated to making deployments of machine learning (ML) workflows on [Kubernetes](https://kubernetes.io/) simple, portable and scalable. Please refer to helpful two slides below about [Kubeflow](https://github.com/kubeflow/kubeflow) which were presented on [KubeCon + CloudNativeCon Europe 2018](https://events.linuxfoundation.org/events/kubecon-cloudnativecon-europe-2018/).

- [Keynote: Cloud Native ML on Kubernetes - David Aronchick, Product Manager, Cloud AI and Co-Founder of Kubeflow, Google & Vishnu Kannan, Sr. Software Engineer, Google](http://sched.co/Duoq)
- [Kubeflow Deep Dive – David Aronchick & Jeremy Lewi, Google](http://sched.co/Drnd)

In this article,  I would like to explain how to run [ChainerMN](https://github.com/chainer/chainermn/) workloads on [Kubernetes](https://kubernetes.io/) with the help of [Kubeflow](https://github.com/kubeflow/kubeflow).

## How to run ChainerMN on Kubernetes
I explain it in three steps below:

- [Step 1. Build Your Container Image](#step-1-build-your-docker-image)
- [Step 2. Install Kubeflow's OpenMPI package](#step-2-install-kubeflows-openmpi-package)
- [Step 3. Run ChainerMN on Kubernetes](step-3-run-chainermn-on-kubernetes)

### Prerequisites
- [Kubernetes](https://kubernetes.io/) cluster equipped with Nvidia GPUs
- on your local machine
  - [docker](https://www.docker.com/community-edition)
  - [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/)
  - [ksonnnet](https://ksonnet.io/)

### Step 1. Build Your Container Image

First we need to build a container image to run your deep learning workload with ChainerMN. All we can just follow [the official ChainerMN installation guides](http://chainermn.readthedocs.io/en/stable/installation/index.html).

For [Chainer](https://chainer.org/)/[Cupy](https://cupy.chainer.org/), official docker image [`chainer/chainer`](https://hub.docker.com/r/chainer/chainer/) is available on DockerHub.  This is very handy as a base image or runtime image for deep learning workloads because this image is already `nvidia-docker` ready.

Below is a sample `Dockerfile` to install CUDA aware [OpenMPI](https://www.open-mpi.org/), [NCCL](https://developer.nvidia.com/nccl), [ChainerMN](https://github.com/chainer/chainermn) and its sample `train_mnist.py` script.  Please save the contents with the name `Dockerfile`.

```
FROM chainer/chainer:v4.0.0-python3

ARG OPENMPI_VERSION="2.1.3"
ARG CHAINER_MN_VERSION="1.2.0"

# Install basic dependencies and locales
RUN apt-get update && apt-get install -yq --no-install-recommends \
      locales wget sudo ca-certificates ssh build-essential && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen

# Install OpenMPI with cuda
RUN cd /tmp && \
  wget -q https://www.open-mpi.org/software/ompi/v${OPENMPI_VERSION%\.*}/downloads/openmpi-$OPENMPI_VERSION.tar.bz2 && \
  tar -xjf openmpi-$OPENMPI_VERSION.tar.bz2 && \
  cd /tmp/openmpi-$OPENMPI_VERSION && \
  ./configure --prefix=/usr --with-cuda && make -j2 && make install && rm -r /tmp/openmpi-$OPENMPI_VERSION* && \
  ompi_info --parsable --all | grep -q "mpi_built_with_cuda_support:value:true"

# Install ChainerMN
RUN pip3 install chainermn==$CHAINER_MN_VERSION

# Download train_mnist.py example of ChainerMN
# In practice, you would download your codes here.
RUN mkdir -p /chainermn-examples/mnist && \
  cd /chainermn-examples/mnist && \
  wget https://raw.githubusercontent.com/chainer/chainermn/v${CHAINER_MN_VERSION}/examples/mnist/train_mnist.py
```

Then, you are ready to build and publish your container image.

```
# This takes some time (probably 10-15 min.). please enjoy ☕️.
docker build . -t YOUR_IMAGE_HERE
docker publish YOUR_IMAGE_HERE
```

### Step 2. Install Kubeflow's OpenMPI package

[Kubeflow's OpenMPI package](https://github.com/kubeflow/kubeflow/tree/master/kubeflow/openmpi/) in [Kubeflow](https://github.com/kubeflow/kubeflow) enables us launch [OpenMPI](https://www.open-mpi.org/) cluster on [Kubernetes](https://kubernetes.io/) very easily.

Actually, __[Kubeflow's OpenMPI package](https://github.com/kubeflow/kubeflow/blob/master/kubeflow/openmpi) have not been released officially__.  But it has been already available in `master` branch of [Kubeflow](https://github.com/kubeflow/kubeflow) repository.  So, Let's use it.  Please note that this package is still in development mode.

Kubeflow depends on [ksonnet](https://ksonnet.io/).  If you're not faimiliar with [ksonnet](https://ksonnet.io/), I recommend you to follow [their official tutorial](https://ksonnet.io/docs/tutorial).

Steps are very similar as discribed in [Kubeflow's OpenMPI package](https://github.com/kubeflow/kubeflow/blob/master/kubeflow/openmpi/).  I modified the original steps slightly because we have to use a specific commit of [Kubeflow](https://github.com/kubeflow/kubeflow) repository.

_NOTE: If you faced [rate limit errors](https://developer.github.com/v3/#rate-limiting) of github api, please set up `GITHUB_TOKEN` as described [here](https://github.com/kubeflow/kubeflow#github-tokens)._

```
# Create a namespace for kubeflow deployment.
NAMESPACE=kubeflow
kubectl create namespace ${NAMESPACE}

# Generate one-time ssh keys used by Open MPI.
SECRET=openmpi-secret
mkdir -p .tmp
yes | ssh-keygen -N "" -f .tmp/id_rsa
kubectl delete secret ${SECRET} -n ${NAMESPACE} || true
kubectl create secret generic ${SECRET} -n ${NAMESPACE} --from-file=id_rsa=.tmp/id_rsa --from-file=id_rsa.pub=.tmp/id_rsa.pub --from-file=authorized_keys=.tmp/id_rsa.pub

# Which version of Kubeflow to use.
# For a list of releases refer to:
# https://github.com/kubeflow/kubeflow/releases
# (Specific commit hash is specified here.)
VERSION=ddaf5298a4cc32cb3834b65150a3281b62c2b49d

# Initialize a ksonnet app. Set the namespace for it's default environment.
APP_NAME=chainermn-example
ks init ${APP_NAME}
cd ${APP_NAME}
ks env set default --namespace ${NAMESPACE}

# Install Kubeflow components.
ks registry add kubeflow github.com/kubeflow/kubeflow/tree/${VERSION}/kubeflow
ks pkg install kubeflow/openmpi@${VERSION}
```

### Step 3. Run ChainerMN!
Now ready to run distributed `train_mnist.py`!  According to standard [ksonnet](https://ksonnet.io/) way, we firstly generate _`train_mnist` component_ from _`openmpi` prototype_.

When generating a component, we can specify several _parameters_.  In this example, we specify

- `train-mnist` for its name,
- `4` workers,
- `1` GPU for each worker, and
- launching `mpiexec ... train_mnist.py` scirpt for `exec` param

And then, `ks apply` command deploy our [OpenMPI](https://www.open-mpi.org/)  cluster on [Kubernetes](https://kubernetes.io/) cluster.

_Please be advised that this step requires an authorization to create service accounts and cluster role bindings for "view" cluster role.  If you didn't have such authorization, you will have to ask your administrator to create a service account which is granted 'get' verb for 'pods' resources. If such service account was ready, you then will set it to `serviceAccountName` param of `train-mnist` component._

```
# See the list of supported parameters.
ks prototype describe openmpi

# Generate openmpi components.
COMPONENT=train-mnist
IMAGE=YOUR_IMAGE_HERE
WORKERS=4
GPU=1
EXEC="mpiexec -n ${WORKERS} --hostfile /kubeflow/openmpi/assets/hostfile --allow-run-as-root --display-map -- python3 /chainermn-examples/mnist/train_mnist.py -g"
ks generate openmpi ${COMPONENT} --image ${IMAGE} --secret ${SECRET} --workers ${WORKERS} --gpu ${GPU} --exec "${EXEC}"

# Deploy to your cluster.
ks apply default

# Clean up, execute below two commands
# ks delete default
# kubectl delete secret ${SECRET}
```

This launches `1` master pod and `4` worker pods and some supplemental parts.  Once `train-mnist-master` pod became `Running` state, training logs will be seen.

```
# Inspect pods status
# Wait until all pods are 'Running'
kubectl get pod -n ${NAMESPACE} -o wide
```

If all went good, our job progress will be seen on your terminal with `kubectl logs`!!  It will show our deep learning jobs are distributed across `4` workers!

```
# Inspect training logs
kubectl logs -n ${NAMESPACE} -f ${COMPONENT}-master
```

This will show you training logs (I omitted several warning messages you can ignore)!!
```
...
========================   JOB MAP   ========================

Data for node: train-mnist-worker-0.train-mnist.kubeflow Num slots: 16   Max slots: 0    Num procs: 1
       Process OMPI jobid: [13015,1] App: 0 Process rank: 0 Bound: N/A

Data for node: train-mnist-worker-1.train-mnist.kubeflow Num slots: 16   Max slots: 0    Num procs: 1
       Process OMPI jobid: [13015,1] App: 0 Process rank: 1 Bound: N/A

Data for node: train-mnist-worker-2.train-mnist.kubeflow Num slots: 16   Max slots: 0    Num procs: 1
       Process OMPI jobid: [13015,1] App: 0 Process rank: 2 Bound: N/A

Data for node: train-mnist-worker-3.train-mnist.kubeflow Num slots: 16   Max slots: 0    Num procs: 1
       Process OMPI jobid: [13015,1] App: 0 Process rank: 3 Bound: N/A

=============================================================
==========================================
Num process (COMM_WORLD): 4
Using GPUs
Using hierarchical communicator
Num unit: 1000
Num Minibatch-size: 100
Num epoch: 20
==========================================
epoch       main/loss   validation/main/loss  main/accuracy  validation/main/accuracy  elapsed_time
1           0.285947    0.106961              0.917333       0.9681                    16.6241
2           0.0870434   0.0882483             0.9736         0.9708                    23.0874
3           0.050553    0.0709311             0.9842         0.9781                    28.6014
...
```
