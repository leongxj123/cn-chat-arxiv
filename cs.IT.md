# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Uncertainty, Calibration, and Membership Inference Attacks: An Information-Theoretic Perspective](https://arxiv.org/abs/2402.10686) | 通过信息论框架分析了最先进的似然比攻击对不确定性、校准水平和数据集大小的影响，研究了成员推理攻击中隐含的风险 |
| [^2] | [Resilience of the quadratic Littlewood-Offord problem](https://arxiv.org/abs/2402.10504) | 论文研究了二次Littlewood-Offord问题的统计鲁棒性，估计了对抗性噪声对二次Radamecher混沌的影响，并提供了对二次和双线性Rademacher混沌的统计鲁棒性的下限估计。 |

# 详细

[^1]: 不确定性、校准和成员推理攻击：信息论视角

    Uncertainty, Calibration, and Membership Inference Attacks: An Information-Theoretic Perspective

    [https://arxiv.org/abs/2402.10686](https://arxiv.org/abs/2402.10686)

    通过信息论框架分析了最先进的似然比攻击对不确定性、校准水平和数据集大小的影响，研究了成员推理攻击中隐含的风险

    

    在成员推理攻击（MIA）中，攻击者利用典型机器学习模型表现出的过度自信来确定特定数据点是否被用于训练目标模型。在本文中，我们在一个信息理论框架内分析了最先进的似然比攻击（LiRA）的性能，这个框架可以允许研究真实数据生成过程中的不确定性的影响，由有限训练数据集引起的认知不确定性以及目标模型的校准水平。我们比较了三种不同的设置，其中攻击者从目标模型接收到的信息逐渐减少：置信向量（CV）披露，其中输出概率向量被发布；真实标签置信度（TLC）披露，其中只有模型分配给真实标签的概率是可用的；以及决策集（DS）披露。

    arXiv:2402.10686v1 Announce Type: cross  Abstract: In a membership inference attack (MIA), an attacker exploits the overconfidence exhibited by typical machine learning models to determine whether a specific data point was used to train a target model. In this paper, we analyze the performance of the state-of-the-art likelihood ratio attack (LiRA) within an information-theoretical framework that allows the investigation of the impact of the aleatoric uncertainty in the true data generation process, of the epistemic uncertainty caused by a limited training data set, and of the calibration level of the target model. We compare three different settings, in which the attacker receives decreasingly informative feedback from the target model: confidence vector (CV) disclosure, in which the output probability vector is released; true label confidence (TLC) disclosure, in which only the probability assigned to the true label is made available by the model; and decision set (DS) disclosure, in 
    
[^2]: 二次Littlewood-Offord问题的弹性

    Resilience of the quadratic Littlewood-Offord problem

    [https://arxiv.org/abs/2402.10504](https://arxiv.org/abs/2402.10504)

    论文研究了二次Littlewood-Offord问题的统计鲁棒性，估计了对抗性噪声对二次Radamecher混沌的影响，并提供了对二次和双线性Rademacher混沌的统计鲁棒性的下限估计。

    

    我们研究了高维数据的统计鲁棒性。我们的结果提供了关于对抗性噪声对二次Radamecher混沌$\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi}$反集中特性的影响的估计，其中$M$是一个固定的（高维）矩阵，$\boldsymbol{\xi}$是一个共形Rademacher向量。具体来说，我们探讨了$\boldsymbol{\xi}$能够承受多少对抗性符号翻转而不“膨胀”$\sup_{x\in \mathbb{R}} \mathbb{P} \left\{\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi} = x\right\}$，从而“去除”原始分布导致更“有粒度”和对抗性偏倚的分布。我们的结果为二次和双线性Rademacher混沌的统计鲁棒性提供了下限估计；这些结果在关键区域被证明是渐近紧的。

    arXiv:2402.10504v1 Announce Type: cross  Abstract: We study the statistical resilience of high-dimensional data. Our results provide estimates as to the effects of adversarial noise over the anti-concentration properties of the quadratic Radamecher chaos $\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi}$, where $M$ is a fixed (high-dimensional) matrix and $\boldsymbol{\xi}$ is a conformal Rademacher vector. Specifically, we pursue the question of how many adversarial sign-flips can $\boldsymbol{\xi}$ sustain without "inflating" $\sup_{x\in \mathbb{R}} \mathbb{P} \left\{\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi} = x\right\}$ and thus "de-smooth" the original distribution resulting in a more "grainy" and adversarially biased distribution. Our results provide lower bound estimations for the statistical resilience of the quadratic and bilinear Rademacher chaos; these are shown to be asymptotically tight across key regimes.
    

