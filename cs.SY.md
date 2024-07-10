# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bucketized Active Sampling for Learning ACOPF.](http://arxiv.org/abs/2208.07497) | 本文提出了一种新颖的主动学习框架——分桶主动采样（BAS），旨在在时间限制内训练最佳的OPF代理。BAS将输入分布分成桶，并使用收集函数确定下一次采样的位置。实验结果显示了BAS的好处。 |

# 详细

[^1]: 学习ACOPF的分桶主动采样

    Bucketized Active Sampling for Learning ACOPF. (arXiv:2208.07497v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2208.07497](http://arxiv.org/abs/2208.07497)

    本文提出了一种新颖的主动学习框架——分桶主动采样（BAS），旨在在时间限制内训练最佳的OPF代理。BAS将输入分布分成桶，并使用收集函数确定下一次采样的位置。实验结果显示了BAS的好处。

    

    本文考虑最优潮流（OPF）的优化代理，即近似OPF的输入/输出关系的机器学习模型。最近的研究集中在证明这些代理可以具有较高的准确性。然而，它们的训练需要大量的数据，每个实例都需要对输入分布的样本进行OPF的（离线）求解。为了满足市场清算应用的要求，本文提出了一种新颖的主动学习框架——分桶主动采样（BAS），旨在在时间限制内训练最佳的OPF代理。BAS将输入分布分成桶，并使用收集函数确定下一次采样的位置。通过将相同的分桶应用于验证集，BAS利用标记的验证样本来选择未标记的样本。BAS还依赖于随时间增加和减少的自适应学习率。实验结果显示了BAS的好处。

    This paper considers optimization proxies for Optimal Power Flow (OPF), i.e., machine-learning models that approximate the input/output relationship of OPF. Recent work has focused on showing that such proxies can be of high fidelity. However, their training requires significant data, each instance necessitating the (offline) solving of an OPF for a sample of the input distribution. To meet the requirements of market-clearing applications, this paper proposes Bucketized Active Sampling (BAS), a novel active learning framework that aims at training the best possible OPF proxy within a time limit. BAS partitions the input distribution into buckets and uses an acquisition function to determine where to sample next. By applying the same partitioning to the validation set, BAS leverages labeled validation samples in the selection of unlabeled samples. BAS also relies on an adaptive learning rate that increases and decreases over time. Experimental results demonstrate the benefits of BAS.
    

