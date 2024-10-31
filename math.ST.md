# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Zeroth-Order Sampling Methods for Non-Log-Concave Distributions: Alleviating Metastability by Denoising Diffusion](https://arxiv.org/abs/2402.17886) | 本文提出了一种基于去噪扩散过程的零阶扩散蒙特卡洛算法，克服了非对数凹分布采样中的亚稳定性问题，并证明其采样精度具有倒多项式依赖。 |

# 详细

[^1]: 用于非对数凹分布的零阶采样方法：通过去噪扩散缓解亚稳定性

    Zeroth-Order Sampling Methods for Non-Log-Concave Distributions: Alleviating Metastability by Denoising Diffusion

    [https://arxiv.org/abs/2402.17886](https://arxiv.org/abs/2402.17886)

    本文提出了一种基于去噪扩散过程的零阶扩散蒙特卡洛算法，克服了非对数凹分布采样中的亚稳定性问题，并证明其采样精度具有倒多项式依赖。

    

    这篇论文考虑了基于其非对数凹分布未归一化密度查询的采样问题。首先描述了一个基于模拟去噪扩散过程的框架，即扩散蒙特卡洛（DMC），其得分函数通过通用蒙特卡洛估计器逼近。DMC是一个基于神谕的元算法，其中神谕是假设可以访问生成蒙特卡洛分数估计器的样本的访问。然后，我们提供了一个基于拒绝采样的这个神谕的实现，这将DMC转化为一个真正的算法，称为零阶扩散蒙特卡洛（ZOD-MC）。我们通过首先构建一个通用框架，即DMC的性能保证，而不假设目标分布为对数凹或满足任何等周不等式，提供了收敛分析。然后我们证明ZOD-MC对所需采样精度具有倒多项式依赖，尽管仍然受到...

    arXiv:2402.17886v1 Announce Type: cross  Abstract: This paper considers the problem of sampling from non-logconcave distribution, based on queries of its unnormalized density. It first describes a framework, Diffusion Monte Carlo (DMC), based on the simulation of a denoising diffusion process with its score function approximated by a generic Monte Carlo estimator. DMC is an oracle-based meta-algorithm, where its oracle is the assumed access to samples that generate a Monte Carlo score estimator. Then we provide an implementation of this oracle, based on rejection sampling, and this turns DMC into a true algorithm, termed Zeroth-Order Diffusion Monte Carlo (ZOD-MC). We provide convergence analyses by first constructing a general framework, i.e. a performance guarantee for DMC, without assuming the target distribution to be log-concave or satisfying any isoperimetric inequality. Then we prove that ZOD-MC admits an inverse polynomial dependence on the desired sampling accuracy, albeit sti
    

