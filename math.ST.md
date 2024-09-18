# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [High-arity PAC learning via exchangeability](https://arxiv.org/abs/2402.14294) | 提出高参数PAC学习理论，利用结构化相关性和交换分布取代i.i.d.抽样，证明了统计学习基本定理的高维版本。 |
| [^2] | [A Dynamical System View of Langevin-Based Non-Convex Sampling.](http://arxiv.org/abs/2210.13867) | 本文提出了一种新的框架，通过利用动力系统理论中的几个工具来解决非凸采样中的重要挑战。对于一大类最先进的采样方案，它们在Wasserstein距离下的最后迭代收敛可以归结为对它们的连续时间对应物的研究，这是更好理解的。 |

# 详细

[^1]: 通过可互换性实现高参数PAC学习

    High-arity PAC learning via exchangeability

    [https://arxiv.org/abs/2402.14294](https://arxiv.org/abs/2402.14294)

    提出高参数PAC学习理论，利用结构化相关性和交换分布取代i.i.d.抽样，证明了统计学习基本定理的高维版本。

    

    我们开发了一种高维PAC学习理论，即在“结构化相关性”存在的统计学习中。 在这个理论中，假设可以是图形、超图，或者更一般地说，是有限关系语言中的结构，并且i.i.d.抽样被抽样产生可互换分布的诱导子结构取代。我们证明了统计学习基本定理的高维版本，通过表征高维（agnostic）PAC可学性，以纯组合维度的有限性及适当版本的均匀收敛。

    arXiv:2402.14294v1 Announce Type: new  Abstract: We develop a theory of high-arity PAC learning, which is statistical learning in the presence of "structured correlation". In this theory, hypotheses are either graphs, hypergraphs or, more generally, structures in finite relational languages, and i.i.d. sampling is replaced by sampling an induced substructure, producing an exchangeable distribution. We prove a high-arity version of the fundamental theorem of statistical learning by characterizing high-arity (agnostic) PAC learnability in terms of finiteness of a purely combinatorial dimension and in terms of an appropriate version of uniform convergence.
    
[^2]: Langevin-Based Non-Convex Sampling的动力学系统视角

    A Dynamical System View of Langevin-Based Non-Convex Sampling. (arXiv:2210.13867v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.13867](http://arxiv.org/abs/2210.13867)

    本文提出了一种新的框架，通过利用动力系统理论中的几个工具来解决非凸采样中的重要挑战。对于一大类最先进的采样方案，它们在Wasserstein距离下的最后迭代收敛可以归结为对它们的连续时间对应物的研究，这是更好理解的。

    This paper proposes a new framework that uses tools from the theory of dynamical systems to address important challenges in non-convex sampling. For a large class of state-of-the-art sampling schemes, their last-iterate convergence in Wasserstein distances can be reduced to the study of their continuous-time counterparts, which is much better understood.

    非凸采样是机器学习中的一个关键挑战，对于深度学习中的非凸优化以及近似概率推断都至关重要。尽管其重要性，理论上仍存在许多重要挑战：现有的保证通常仅适用于平均迭代而不是更理想的最后迭代，缺乏捕捉变量尺度（如Wasserstein距离）的收敛度量，主要适用于随机梯度Langevin动力学等基本方案。在本文中，我们开发了一个新的框架，通过利用动力系统理论中的几个工具来解决上述问题。我们的关键结果是，对于一大类最先进的采样方案，它们在Wasserstein距离下的最后迭代收敛可以归结为对它们的连续时间对应物的研究，这是更好理解的。结合MCMC采样的标准假设，我们的理论立即产生了

    Non-convex sampling is a key challenge in machine learning, central to non-convex optimization in deep learning as well as to approximate probabilistic inference. Despite its significance, theoretically there remain many important challenges: Existing guarantees (1) typically only hold for the averaged iterates rather than the more desirable last iterates, (2) lack convergence metrics that capture the scales of the variables such as Wasserstein distances, and (3) mainly apply to elementary schemes such as stochastic gradient Langevin dynamics. In this paper, we develop a new framework that lifts the above issues by harnessing several tools from the theory of dynamical systems. Our key result is that, for a large class of state-of-the-art sampling schemes, their last-iterate convergence in Wasserstein distances can be reduced to the study of their continuous-time counterparts, which is much better understood. Coupled with standard assumptions of MCMC sampling, our theory immediately yie
    

