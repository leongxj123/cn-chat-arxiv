# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Interference Among First-Price Pacing Equilibria: A Bias and Variance Analysis](https://arxiv.org/abs/2402.07322) | 本文提出了一种并行的预算控制的A/B测试设计，通过市场细分的方式在更大的市场中识别子市场，并在每个子市场上进行并行实验。 |

# 详细

[^1]: 第一价拍卖均衡中的干扰：偏差和方差分析

    Interference Among First-Price Pacing Equilibria: A Bias and Variance Analysis

    [https://arxiv.org/abs/2402.07322](https://arxiv.org/abs/2402.07322)

    本文提出了一种并行的预算控制的A/B测试设计，通过市场细分的方式在更大的市场中识别子市场，并在每个子市场上进行并行实验。

    

    在互联网行业中，在线A/B测试被广泛用于决策新功能的推出。然而对于在线市场（如广告市场），标准的A/B测试方法可能导致结果出现偏差，因为买家在预算约束下运作，试验组的预算消耗会影响对照组的表现。为了解决这种干扰，可以采用“预算分割设计”，即每个实验组都有一个独立的预算约束，并且每个实验组接收相等的预算份额，从而实现“预算控制的A/B测试”。尽管预算控制的A/B测试有明显的优势，但当预算分割得太小时，性能会下降，限制了这种系统的总吞吐量。本文提出了一种并行的预算控制的A/B测试设计，通过市场细分的方式在更大的市场中识别子市场，并在每个子市场上进行并行实验。我们的贡献如下：首先，引入了一种新的方法来分析第一价拍卖的均衡状况，揭示了其中的偏差和方差。

    Online A/B testing is widely used in the internet industry to inform decisions on new feature roll-outs. For online marketplaces (such as advertising markets), standard approaches to A/B testing may lead to biased results when buyers operate under a budget constraint, as budget consumption in one arm of the experiment impacts performance of the other arm. To counteract this interference, one can use a budget-split design where the budget constraint operates on a per-arm basis and each arm receives an equal fraction of the budget, leading to ``budget-controlled A/B testing.'' Despite clear advantages of budget-controlled A/B testing, performance degrades when budget are split too small, limiting the overall throughput of such systems. In this paper, we propose a parallel budget-controlled A/B testing design where we use market segmentation to identify submarkets in the larger market, and we run parallel experiments on each submarket.   Our contributions are as follows: First, we introdu
    

