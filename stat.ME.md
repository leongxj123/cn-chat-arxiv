# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Conditional Generative Models are Sufficient to Sample from Any Causal Effect Estimand](https://arxiv.org/abs/2402.07419) | 本文展示了通过条件生成模型的推进计算可以计算任何可辨识的因果效应，并提出了基于扩散的方法用于从图像的任何（条件）干预分布中进行采样。 |

# 详细

[^1]: 条件生成模型足以从任何因果效应测度中采样

    Conditional Generative Models are Sufficient to Sample from Any Causal Effect Estimand

    [https://arxiv.org/abs/2402.07419](https://arxiv.org/abs/2402.07419)

    本文展示了通过条件生成模型的推进计算可以计算任何可辨识的因果效应，并提出了基于扩散的方法用于从图像的任何（条件）干预分布中进行采样。

    

    最近，从观测数据进行因果推断在机器学习中得到了广泛应用。虽然存在计算因果效应的可靠且完备的算法，但其中许多算法需要显式访问观测分布上的条件似然，而在高维场景中（例如图像），估计这些似然是困难的。为了解决这个问题，研究人员通过使用神经模型模拟因果关系，并取得了令人印象深刻的结果。然而，这些现有方法中没有一个可以应用于通用场景，例如具有潜在混淆因素的图像数据的因果图，或者获得条件干预样本。在本文中，我们展示了在任意因果图下，通过条件生成模型的推进计算可以计算任何可辨识的因果效应。基于此结果，我们设计了一个基于扩散的方法，可以从任何（条件）干预分布中采样图像。

    Causal inference from observational data has recently found many applications in machine learning. While sound and complete algorithms exist to compute causal effects, many of these algorithms require explicit access to conditional likelihoods over the observational distribution, which is difficult to estimate in the high-dimensional regime, such as with images. To alleviate this issue, researchers have approached the problem by simulating causal relations with neural models and obtained impressive results. However, none of these existing approaches can be applied to generic scenarios such as causal graphs on image data with latent confounders, or obtain conditional interventional samples. In this paper, we show that any identifiable causal effect given an arbitrary causal graph can be computed through push-forward computations of conditional generative models. Based on this result, we devise a diffusion-based approach to sample from any (conditional) interventional distribution on ima
    

