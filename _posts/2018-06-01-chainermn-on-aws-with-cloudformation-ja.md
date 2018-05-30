---
title: AWS CloudForamtionを使ったChainerMNの実行
layout: post
categories: General
author: Shingo Omura
---

_English version is [here]({{ site.baseurl }}/general/2018/06/01/chainermn-on-aws-with-cloudformation.html)._

[AWS CloudFormation][CFN]は [_Infrastructure As Code_](https://en.wikipedia.org/wiki/Infrastructure_as_Code) の実践を助けてくれるAWSサービスで、幅広いAWSリソースを、宣言的に設定ファイルに記述し、その設定ファイルから、AWS上のインフラストラクチャを自動的に生成したり、再生成できます。それによってAWS上のインフラストラクチャを手作業を自動化できます。

分散深層学習向けのインフラストラクチャの構築の作業負荷も大幅に軽減できます。EC2インスタンスを起動したり、必要なライブラリをインストール・設定したり、計算/通信性能の最適化設定を行ったりする作業を軽減できます。特に、[ChainerMN][ChainerMN]においては、MPIクラスタを構築することが必要です。AWS CloudFormationはこの構築作業を自動化することができます。

本日、[Chainer/ChainermNがプリインストールされたAMI][ChainerAMI]と[ChainerMN向けのクラスタを自動的に構築するためのCloudFormationテンプレート[ChainerCFN]を公開しました。

- [chainer/chainer-ami][ChainerAMI]
- [chainer/chainer-cfn][ChainerCFN]

この記事では、これらの利用方法とともに、自動構築されたクラスタ上での[ChainerMN][ChainerMN]の実行方法について説明します。

[Chainer AMI][ChainerAMI]
-----
[Chainer AMI][ChainerAMI] には [Chainer][Chainer]/[CuPy][CuPy]/[ChainerMN][ChainerMN], [ChianerCV][ChainerCV], [ChainerRL][ChainerRL]といったChainerライブラリ群と [CUDA][CUDA]-aware [OpenMPI][OpenMPI] ライブラリがプリインストールされていいます。そのため、このイメージを利用すればGPUを搭載したAWS EC2 インスタンス上で簡単に高速に深層学習が実行できます。このイメージは[AWS Deep Learning Base AMI](https://docs.aws.amazon.com/dlami/latest/devguide/overview-base.html)を基にビルドされています。

[Chainer AMI][ChainerAMI]の最新バージョンは`0.1.0`で、同梱されているライブラリ群のバージョンは次のとおりです:

- OpenMPI `2.1.3`
  -[CUDA][CUDA] 9向けに __のみ__ ビルドされています
- Chainerライブラリ群 (`python`, `python3` 両方の環境にインストールされています)
  - `CuPy 4.1.0`
  - `Chainer 4.1.0`
  - `ChainerMN 1.3.0`
  - `ChainerCV 0.9.0`
  - `ChainerRL 0.3.0`

[CloudFormation Template For ChainerMN][ChainerCFN]
---
このテンプレートは [ChainerMN][ChainerMN] クラスタを自動的にAWS上に構築します。構築されるインフラストラクチャの概要は下記のとおりです。

- クラスタが配置される VPC と Subnet (既存のVPC/Subnetを設定可能)
- MPIプロセス起動時に利用するephemeralなssh鍵をクラスタ内で共有するための S3 バケット
- クラスタが配置されるプレイスメントグループ(ネットワークスループット/レイテンシを最適化するため)
- ChainerMNクラスタ
  - `1` 台の マスター EC2 インスタンス
  - `N (>=0)` 台のワーカーインスタンス(AutoScalingGroup経由で起動されます)
  - MPIジョブ実行用の `chainer` ユーザ
  - MPIジョブ実行用の `hostfile`
- (オプション) [Amazon Elastic Filesystem][EFS] (既存のFilesystemを設定可能)
  - このファイルシステムは各インスタンスに自動的にマウントされれます
- 各種Security Group および IAM Role


[Chaainer CFN][ChainerCFN]の最新バージョンは`0.1.0`です。詳細なリソース定義については[リリースされているテンプレート](https://s3-us-west-2.amazonaws.com/chainer-cfn/chainer-cfn-v0.1.0.template)を参照してください。

[先日公開した ChainerMN `1.3.0` のブログで述べられている通り(英語)](https://chainer.org/general/2018/05/25/chainermn-v1-3.html), [ChainerMN][ChainerMN] `1.3.0` の新機能である、ダブルバッファリング、 半精度浮動小数点数による All-Reduce を使えば、Infinibandが利用できないAWSであっても、ほぼ線形のスケーラビリティを期待できます。

[CloudFromationテンプレート][ChainerCFN] を使った [ChainerMN][ChainerMN] クラスタ構築
---

ここでは、[CloudFromationテンプレート][ChainerCFN]を使って[ChainerMN][ChainerMN]クラスタを構築する方法をstep-by-stepで説明します。

まず、下記のリンクから CloudFormationのページを開いて「次へ」をクリックしてください。

[![launch stack](https://s3.amazonaws.com/cloudformation-examples/cloudformation-launch-stack.png)](https://console.aws.amazon.com/cloudformation/home#/stacks/new?stackName=chainermn-sample&templateURL=https://s3-us-west-2.amazonaws.com/chainer-cfn/chainer-cfn-v0.1.0.template)

「詳細の指定」画面では、スタック名、VPC/Subnet、クラスタ、EFSといった各種設定を入力できます。下記は `4` 台の `p3.16xlarge`インスタンス(8 NVIDIA Tesla V100 GPUs/インスタンス) によって構成される ChainerMNクラスタを構築する設定例です(VPC, Subnet, EFS等はすべて新規作成)。

![chainer-cfn-specifying-details]({{ site.baseurl }}/images/chainer-cfn-specifying-details.png)

最後の確認画面では、`CAPABILITY` セクションにある、IAM Roleに関する同意をチェックする必要があります。この CloudFormation テンプレートではクラスタ内インスタンスに割り当てるためのIAM Roleを作成するためです。

![chainer-cfn-specifying-details]({{ site.baseurl }}/images/chainer-cfn-capabilities-confirmation.png)

しばらして、CloudFormationスタックの状態が `CREATE_COMPLETE` に遷移したら、クラスタは構築完了です。 スタックの出力セクションの `ClusterMasterPublicDNS` にマスターインスタンスの Public DNS が表示されます。

構築済みクラスタでの [ChainerMN][ChainerMN] ジョブ実行
--
クラスタ内インスタンスには、CloudFormationテンプレートのパラメータに設定したキーペアでログインが可能です。

```
ssh -i keypair.pem ubuntu@ec2-ww-xxx-yy-zzz.compute-1.amazonaws.com
```

クラスタ内のインスタンスは前述の [Chainer AMI][ChainerAMI] をつかって起動しているので、必要なライブラリ群はすべてインストール済みです。あとは、自身の学習用のコードとデータをダウンロードするだけです。

```
# switch user to chainer
ubuntu@ip-ww-xxx-yy-zzz$ sudo su chainer

# download ChainerMN's train_mnist.py into EFS
chainer@ip-ww-xxx-yy-zzz$ wget https://raw.githubusercontent.com/chainer/chainermn/v1.3.0/examples/mnist/train_mnist.py -O /efs/train_mnist.py
```

これで実行準備完了です。`mpiexec`コマンドを使って[ChainerMN][ChainerMN]をつかったMNISTの例を実行できます。

```
# 4インスタンス上で合計32プロセス(-nオプション)を起動(8 プロセス/インスタンス(-N オプション))

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

# 注意: 上記実行例は２回めの実行例です。(初回実行時はサンプルデータのダウンロードが行われます)
```


[CFN]: https://aws.amazon.com/jp/cloudformation/
[EFS]: https://aws.amazon.com/jp/efs/features/
[ChainerAMI]: https://github.com/chainer/chainer-ami
[ChainerCFN]: https://github.com/chainer/chainer-cfn
[ChainerMN]: https://github.com/chainer/chainermn
[Chainer]: https://chainer.org
[CuPy]: https://cupy.chainer.org/
[ChainerCV]: https://github.com/chainer/chainercv
[ChainerRL]: https://github.com/chainer/chainerrl
[CUDA]: https://developer.nvidia.com/cuda-zone
[OpenMPI]: https://www.open-mpi.org/
