# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Causal Inference with Differentially Private (Clustered) Outcomes.](http://arxiv.org/abs/2308.00957) | 本文提出了一种新的差分隐私机制"Cluster-DP"，它在保证隐私的同时利用数据的聚类结构，从而实现了更强的隐私保证和较低的方差，可以用于进行因果分析。 |

# 详细

[^1]: 具有差分隐私(分组)结果的因果推断

    Causal Inference with Differentially Private (Clustered) Outcomes. (arXiv:2308.00957v1 [stat.ML])

    [http://arxiv.org/abs/2308.00957](http://arxiv.org/abs/2308.00957)

    本文提出了一种新的差分隐私机制"Cluster-DP"，它在保证隐私的同时利用数据的聚类结构，从而实现了更强的隐私保证和较低的方差，可以用于进行因果分析。

    

    从随机实验中估计因果效应只有在参与者同意透露他们可能敏感的响应时才可行。在确保隐私的许多方法中，标签差分隐私是一种广泛使用的算法隐私保证度量，可以鼓励参与者分享响应而不会面临去匿名化的风险。许多差分隐私机制会向原始数据集中注入噪音来实现这种隐私保证，这会增加大多数统计估计量的方差，使得精确测量因果效应变得困难：从差分隐私数据进行因果分析存在着固有的隐私-方差权衡。为了实现更强隐私保证的较低方差，我们提出了一种新的差分隐私机制"Cluster-DP"，它利用数据的任何给定的聚类结构，同时仍然允许对因果效应进行估计。

    Estimating causal effects from randomized experiments is only feasible if participants agree to reveal their potentially sensitive responses. Of the many ways of ensuring privacy, label differential privacy is a widely used measure of an algorithm's privacy guarantee, which might encourage participants to share responses without running the risk of de-anonymization. Many differentially private mechanisms inject noise into the original data-set to achieve this privacy guarantee, which increases the variance of most statistical estimators and makes the precise measurement of causal effects difficult: there exists a fundamental privacy-variance trade-off to performing causal analyses from differentially private data. With the aim of achieving lower variance for stronger privacy guarantees, we suggest a new differential privacy mechanism, "Cluster-DP", which leverages any given cluster structure of the data while still allowing for the estimation of causal effects. We show that, depending 
    

