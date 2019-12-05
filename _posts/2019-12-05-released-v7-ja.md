---
title: Chainer/CuPy v7のリリースと今後の開発体制について
layout: post
categories: Announcement
author: Chainer Team
---

Chainer/CuPy v7のリリース、およびChainerの開発体制の変更についてお知らせします。

## Chainer/CuPy v7

本日、ChainerおよびCuPyのv7.0.0をリリースしました。変更点については各リリースノートをご覧ください。主要な変更点は以下の通りです。

Chainer v7 ([alpha](https://github.com/chainer/chainer/releases/tag/v7.0.0a1), [beta1](https://github.com/chainer/chainer/releases/tag/v7.0.0b1), [beta2](https://github.com/chainer/chainer/releases/tag/v7.0.0b2), [beta3](https://github.com/chainer/chainer/releases/tag/v7.0.0b3), [beta4](https://github.com/chainer/chainer/releases/tag/v7.0.0b4), [rc1](https://github.com/chainer/chainer/releases/tag/v7.0.0rc1), [major](https://github.com/chainer/chainer/releases/tag/v7.0.0)):

- ChainerMNを含む多くの機能がChainerXのndarrayに対応しました。
- ONNX-ChainerがChainerに統合されました。
- `TabularDataset` が追加されました。カラム指向のデータセットをpandasのような抽象化APIで操作できます。
- NHWCのサポートが追加されました。Tensor Coreを搭載したGPUにおいて畳み込みやBatch Normalizationのパフォーマンスが向上します。

CuPy v7 ([alpha](https://github.com/cupy/cupy/releases/tag/v7.0.0a1), [beta1](https://github.com/cupy/cupy/releases/tag/v7.0.0b1), [beta2](https://github.com/cupy/cupy/releases/tag/v7.0.0b2), [beta3](https://github.com/cupy/cupy/releases/tag/v7.0.0b3), [beta4](https://github.com/cupy/cupy/releases/tag/v7.0.0b4), [rc1](https://github.com/cupy/cupy/releases/tag/v7.0.0rc1), [major](https://github.com/cupy/cupy/releases/tag/v7.0.0)):

- NVIDIA cuTENSORおよびCUBのサポートによりパフォーマンスが向上しました。
- ROCmの試験的なサポートを行いました。これにより、CuPyがAMD GPU上で実行可能になります。

なお、すでに[アナウンス](https://chainer.org/announcement/2019/08/21/python2.html)した通り、Python 2のサポートが終了しました。Chainer/CuPy v7ではPython 3.5以降のみがサポートされます。

## Chainer開発体制の変更について

本日[アナウンス](https://preferred.jp/ja/news/pr20191205/)された通り、Chainerの開発元であるPreferred Networksでは、研究開発に使用するフレームワークをPyTorchへ順次移行します。現時点では、Chainer v7はChainerの最後のメジャーリリースとなる予定であり、今後の開発はバグフィックスおよびメンテナンスのみとなります。Chainerファミリー(ChainerCV, Chainer Chemistry, ChainerUI, ChainerRL)についてもこの方針に従います。また、Preferred Networksの運用する[ディープラーニング入門: Chainerチュートリアル](https://tutorials.chainer.org/ja/)については今後コンテンツのリニューアルを検討しています。

なお、CuPyの開発はこれまで通り継続してゆきます。CuPyは当初ChainerのGPUバックエンドとして開発されましたが、現在ではGPUによる高速な演算をNumPyと同じ文法で記述できる数少ないライブラリとして、様々なコミュニティに受け入れられています。

### 背景

この決定は、「深層学習およびその応用の研究開発を高速化する」というChainerチームのミッションを踏まえ、様々な検討を重ねた上で慎重に行われました。

2015年に公開されたChainerは、微分可能プログラミングのための新たな命令的APIセットを提案し、それを *define-by-run* と名付けました。このパラダイムは、今日では *eager* executionとも呼ばれています。当初define-by-runのアプローチは、自然言語処理に用いられる回帰型ニューラルネットワーク(RNN)などの記述を容易にするというモチベーションから発案されたものでしたが、すぐにそれ以外のネットワークにも応用されてゆきました。その直感的な表記とデバッグの容易さは、深層学習研究における開発サイクルの高速化に大きく貢献しました。我々は命令的な実行方式を採用するフレームワークが、既存の宣言的な *define-and-run* 実行方式よりも優れているという確信を得て、開発を進めました。オブジェクト指向によるネットワーク定義、高次微分、レイヤの入力データサイズの動的推論、トレーニングループの抽象化といった様々な機能追加を、pure Pythonによる簡潔な実装とNumPyエコシステムとの相互運用性を保ったまま実現してきました。

define-by-runのアプローチは深層学習コミュニティにおいて広く受け入れられ、結果として多くのフレームワークは似通った文法と機能に集約されてゆきました。Chainerチームは、このトレンドの転換においてChainerが果たした役割を誇りに思うとともに、コミュニティに対してこのような貢献ができたことを嬉しく思います。そして今、研究開発の生産性を高めるために深層学習コミュニティに対してどのような貢献をしてゆくべきか改めて熟慮した結果、似通ったゴールを持つフレームワークを個別に開発するのではなく、より大きなユーザベースとエコシステムを持つフレームワークに貢献してゆくことが最良であると判断しました。

いくつかのフレームワークを検討したのち、PyTorchが最もChainerに近い思想を持っており、Chainerの後続として最適であると確信しました。Preferred Networksでは、今後PyTorchを主要なフレームワークとして使用するとともに、Chainerの開発を通じて得られた知識と経験を生かしてPyTorchへ貢献してゆきます。

### おわりに

PyTorchへの移行に際して、Chainerチームでは移行を容易にするためのドキュメントおよびライブラリを公開しました。

- [Migration Guide](http://chainer.github.io/migration-guide)
- [Migration Library](http://github.com/chainer/chainer-pytorch-migration)

これまでChainerおよびChainerを取り巻くコミュニティへ貢献してくださった全ての皆さまに、深く感謝いたします。今日の成果は、皆さまの協力なくして成し得ませんでした。今後も深層学習ソフトウェアの改善を通じて、コミュニティと協働しながら深層学習領域の研究開発の加速に貢献してゆきたいと考えています。

[英語版 (English)](https://chainer.org/announcement/2019/12/05/released-v7.html)
