# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [How Much Does Each Datapoint Leak Your Privacy? Quantifying the Per-datum Membership Leakage](https://arxiv.org/abs/2402.10065) | 本论文研究了每个数据点的成员推断攻击，量化了每个数据点的成员泄露，并评估了两种隐私防御措施的效果。 |
| [^2] | [$\sigma$-zero: Gradient-based Optimization of $\ell_0$-norm Adversarial Examples](https://arxiv.org/abs/2402.01879) | 该论文提出了一种新的基于梯度的$\ell_0$范数攻击方法$\sigma$-zero，其利用了$\ell_0$范数的可微近似和自适应投影运算符，能够在非凸和非可微的约束下优化，从而评估深度网络对稀疏$\ell_0$范数攻击的鲁棒性。 |

# 详细

[^1]: 每个数据点泄露您隐私的程度有多大？量化每个数据点的成员泄露

    How Much Does Each Datapoint Leak Your Privacy? Quantifying the Per-datum Membership Leakage

    [https://arxiv.org/abs/2402.10065](https://arxiv.org/abs/2402.10065)

    本论文研究了每个数据点的成员推断攻击，量化了每个数据点的成员泄露，并评估了两种隐私防御措施的效果。

    

    我们研究了每个数据点的成员推断攻击（MIAs），其中攻击者旨在推断出一个固定目标数据是否已包含在算法的输入数据集中，从而侵犯隐私。首先，我们定义每个数据点的成员泄露为最优对手辨识它的优势。然后，我们量化了经验均值的每个数据点的成员泄露，并表明它取决于目标数据点和数据生成分布之间的马氏距离。我们进一步评估了两种隐私防御措施的效果，即添加高斯噪声和子采样。我们准确地量化了它们都如何降低每个数据点的成员泄露。我们的分析建立在一个结合了似然比检验的Edgeworth展开和Lindeberg-Feller中心极限定理的新型证明技术上。我们的分析连接了现有的似然比和标量乘积攻击，并对这些攻击进行了论证。

    arXiv:2402.10065v1 Announce Type: new  Abstract: We study the per-datum Membership Inference Attacks (MIAs), where an attacker aims to infer whether a fixed target datum has been included in the input dataset of an algorithm and thus, violates privacy. First, we define the membership leakage of a datum as the advantage of the optimal adversary targeting to identify it. Then, we quantify the per-datum membership leakage for the empirical mean, and show that it depends on the Mahalanobis distance between the target datum and the data-generating distribution. We further assess the effect of two privacy defences, i.e. adding Gaussian noise and sub-sampling. We quantify exactly how both of them decrease the per-datum membership leakage. Our analysis builds on a novel proof technique that combines an Edgeworth expansion of the likelihood ratio test and a Lindeberg-Feller central limit theorem. Our analysis connects the existing likelihood ratio and scalar product attacks, and also justifies 
    
[^2]: $\sigma$-zero: 基于梯度的$\ell_0$-范数对抗样本优化

    $\sigma$-zero: Gradient-based Optimization of $\ell_0$-norm Adversarial Examples

    [https://arxiv.org/abs/2402.01879](https://arxiv.org/abs/2402.01879)

    该论文提出了一种新的基于梯度的$\ell_0$范数攻击方法$\sigma$-zero，其利用了$\ell_0$范数的可微近似和自适应投影运算符，能够在非凸和非可微的约束下优化，从而评估深度网络对稀疏$\ell_0$范数攻击的鲁棒性。

    

    评估深度网络对基于梯度攻击的对抗鲁棒性是具有挑战性的。虽然大多数攻击考虑$\ell_2$和$\ell_\infty$范数约束来制造输入扰动，但只有少数研究了稀疏的$\ell_1$和$\ell_0$范数攻击。特别是，由于在非凸且非可微约束上进行优化的固有复杂性，$\ell_0$范数攻击是研究最少的。然而，使用这些攻击评估对抗鲁棒性可以揭示在更传统的$\ell_2$和$\ell_\infty$范数攻击中未能测试出的弱点。在这项工作中，我们提出了一种新颖的$\ell_0$范数攻击，称为$\sigma$-zero，它利用了$\ell_0$范数的一个特殊可微近似来促进基于梯度的优化，并利用自适应投影运算符动态调整损失最小化和扰动稀疏性之间的权衡。通过在MNIST、CIFAR10和ImageNet数据集上进行广泛评估，包括...

    Evaluating the adversarial robustness of deep networks to gradient-based attacks is challenging. While most attacks consider $\ell_2$- and $\ell_\infty$-norm constraints to craft input perturbations, only a few investigate sparse $\ell_1$- and $\ell_0$-norm attacks. In particular, $\ell_0$-norm attacks remain the least studied due to the inherent complexity of optimizing over a non-convex and non-differentiable constraint. However, evaluating adversarial robustness under these attacks could reveal weaknesses otherwise left untested with more conventional $\ell_2$- and $\ell_\infty$-norm attacks. In this work, we propose a novel $\ell_0$-norm attack, called $\sigma$-zero, which leverages an ad hoc differentiable approximation of the $\ell_0$ norm to facilitate gradient-based optimization, and an adaptive projection operator to dynamically adjust the trade-off between loss minimization and perturbation sparsity. Extensive evaluations using MNIST, CIFAR10, and ImageNet datasets, involving
    

