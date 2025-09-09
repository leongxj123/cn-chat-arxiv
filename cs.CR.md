# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards Principled Assessment of Tabular Data Synthesis Algorithms](https://arxiv.org/abs/2402.06806) | 本文提出了一个原则性和系统化的评估框架来评估表格数据合成算法，包括保真度、隐私性和实用性等新指标，以解决现有评估指标的限制。通过这个框架，对不同算法进行了比较和总结。 |
| [^2] | [On Rate-Optimal Partitioning Classification from Observable and from Privatised Data](https://arxiv.org/abs/2312.14889) | 研究了在放宽条件下的分区分类方法的收敛速率，提出了绝对连续分量的新特性，计算了分类错误概率的精确收敛率 |

# 详细

[^1]: 关于表格数据合成算法的原则性评估

    Towards Principled Assessment of Tabular Data Synthesis Algorithms

    [https://arxiv.org/abs/2402.06806](https://arxiv.org/abs/2402.06806)

    本文提出了一个原则性和系统化的评估框架来评估表格数据合成算法，包括保真度、隐私性和实用性等新指标，以解决现有评估指标的限制。通过这个框架，对不同算法进行了比较和总结。

    

    数据合成被认为是一种利用数据同时保护数据隐私的重要方法。已经提出了大量的表格数据合成算法（我们称之为合成器）。一些合成器满足差分隐私，而其他一些则旨在以启发式的方式提供隐私保护。由于缺乏原则性评估指标以及对利用扩散模型和最新的基于边际的合成器与大型语言模型进行面对面比较的新开发的合成器的理解尚不全面，对这些合成器的优势和弱点的全面了解仍然难以实现。在本文中，我们提出了一个原则性和系统化的评估框架来评估表格数据合成算法。具体而言，我们检查和批评现有的评估指标，并引入了一组新的指标，以解决其限制，包括保真度、隐私性和实用性。基于提出的指标，我们还设计了一个统一的评估组织框架，以对不同算法进行评估并进行比较和总结。

    Data synthesis has been advocated as an important approach for utilizing data while protecting data privacy. A large number of tabular data synthesis algorithms (which we call synthesizers) have been proposed. Some synthesizers satisfy Differential Privacy, while others aim to provide privacy in a heuristic fashion. A comprehensive understanding of the strengths and weaknesses of these synthesizers remains elusive due to lacking principled evaluation metrics and missing head-to-head comparisons of newly developed synthesizers that take advantage of diffusion models and large language models with state-of-the-art marginal-based synthesizers.   In this paper, we present a principled and systematic evaluation framework for assessing tabular data synthesis algorithms. Specifically, we examine and critique existing evaluation metrics, and introduce a set of new metrics in terms of fidelity, privacy, and utility to address their limitations. Based on the proposed metrics, we also devise a un
    
[^2]: 论从可观测和私密数据中实现速率最优分区分类

    On Rate-Optimal Partitioning Classification from Observable and from Privatised Data

    [https://arxiv.org/abs/2312.14889](https://arxiv.org/abs/2312.14889)

    研究了在放宽条件下的分区分类方法的收敛速率，提出了绝对连续分量的新特性，计算了分类错误概率的精确收敛率

    

    在这篇论文中，我们重新审视了分区分类的经典方法，并研究了在放宽条件下的收敛速率，包括可观测（非私密）和私密数据。我们假设特征向量$X$取值于$\mathbb{R}^d$，其标签为$Y$。之前关于分区分类器的结果基于强密度假设，这种假设限制较大，我们通过简单的例子加以证明。我们假设$X$的分布是绝对连续分布和离散分布的混合体，其中绝对连续分量集中于一个$d_a$维子空间。在这里，我们在更宽松的条件下研究了这个问题：除了标准的Lipschitz和边际条件外，我们还引入了绝对连续分量的一个新特性，通过该特性计算了分类错误概率的精确收敛率，对于...

    arXiv:2312.14889v2 Announce Type: replace-cross  Abstract: In this paper we revisit the classical method of partitioning classification and study its convergence rate under relaxed conditions, both for observable (non-privatised) and for privatised data. Let the feature vector $X$ take values in $\mathbb{R}^d$ and denote its label by $Y$. Previous results on the partitioning classifier worked with the strong density assumption, which is restrictive, as we demonstrate through simple examples. We assume that the distribution of $X$ is a mixture of an absolutely continuous and a discrete distribution, such that the absolutely continuous component is concentrated to a $d_a$ dimensional subspace. Here, we study the problem under much milder assumptions: in addition to the standard Lipschitz and margin conditions, a novel characteristic of the absolutely continuous component is introduced, by which the exact convergence rate of the classification error probability is calculated, both for the
    

