# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Neural Bayes Estimators for Irregular Spatial Data using Graph Neural Networks.](http://arxiv.org/abs/2310.02600) | 通过使用图神经网络，该论文提出了一种解决非规则空间数据的参数估计问题的方法，扩展了神经贝叶斯估计器的应用范围，并带来了显著的计算优势。 |
| [^2] | [Maximum Mean Discrepancy Meets Neural Networks: The Radon-Kolmogorov-Smirnov Test.](http://arxiv.org/abs/2309.02422) | 本文将最大均差相似度应用于神经网络，并提出了一种称为Radon-Kolmogorov-Smirnov（RKS）检验的方法，该方法将样本均值差异最大化的问题推广到多维空间和更高平滑度顺序，同时与神经网络密切相关。 |

# 详细

[^1]: 使用图神经网络的非规则空间数据的神经贝叶斯估计器

    Neural Bayes Estimators for Irregular Spatial Data using Graph Neural Networks. (arXiv:2310.02600v1 [stat.ME])

    [http://arxiv.org/abs/2310.02600](http://arxiv.org/abs/2310.02600)

    通过使用图神经网络，该论文提出了一种解决非规则空间数据的参数估计问题的方法，扩展了神经贝叶斯估计器的应用范围，并带来了显著的计算优势。

    

    神经贝叶斯估计器是一种以快速和免似然方式逼近贝叶斯估计器的神经网络。它们在空间模型和数据中的使用非常吸引人，因为估计经常是计算上的瓶颈。然而，到目前为止，空间应用中的神经贝叶斯估计器仅限于在规则的网格上收集的数据。这些估计器目前还依赖于预先规定的空间位置，这意味着神经网络需要重新训练以适应新的数据集；这使它们在许多应用中变得不实用，并阻碍了它们的广泛应用。在本研究中，我们采用图神经网络来解决从任意空间位置收集的数据进行参数估计的重要问题。除了将神经贝叶斯估计扩展到非规则空间数据之外，我们的架构还带来了显着的计算优势，因为该估计器可以用于任何排列或数量的位置和独立的重复实验中。

    Neural Bayes estimators are neural networks that approximate Bayes estimators in a fast and likelihood-free manner. They are appealing to use with spatial models and data, where estimation is often a computational bottleneck. However, neural Bayes estimators in spatial applications have, to date, been restricted to data collected over a regular grid. These estimators are also currently dependent on a prescribed set of spatial locations, which means that the neural network needs to be re-trained for new data sets; this renders them impractical in many applications and impedes their widespread adoption. In this work, we employ graph neural networks to tackle the important problem of parameter estimation from data collected over arbitrary spatial locations. In addition to extending neural Bayes estimation to irregular spatial data, our architecture leads to substantial computational benefits, since the estimator can be used with any arrangement or number of locations and independent repli
    
[^2]: 最大均差相似度遇上神经网络：Radon-Kolmogorov-Smirnov检验

    Maximum Mean Discrepancy Meets Neural Networks: The Radon-Kolmogorov-Smirnov Test. (arXiv:2309.02422v1 [stat.ML])

    [http://arxiv.org/abs/2309.02422](http://arxiv.org/abs/2309.02422)

    本文将最大均差相似度应用于神经网络，并提出了一种称为Radon-Kolmogorov-Smirnov（RKS）检验的方法，该方法将样本均值差异最大化的问题推广到多维空间和更高平滑度顺序，同时与神经网络密切相关。

    

    最大均差相似度（MMD）是一类基于最大化两个分布$P$和$Q$之间样本均值差异的非参数双样本检验，其中考虑了所有在某个函数空间$\mathcal{F}$中的数据变换$f$的选择。受到最近将所谓的Radon有界变差函数（RBV）和神经网络联系起来的工作的启发（Parhi和Nowak, 2021, 2023），我们研究了将$\mathcal{F}$取为给定平滑度顺序$k \geq 0$下的RBV空间中的单位球的MMD。这个检验被称为Radon-Kolmogorov-Smirnov（RKS）检验，可以看作是对多维空间和更高平滑度顺序的经典Kolmogorov-Smirnov（KS）检验的一般化。它还与神经网络密切相关：我们证明RKS检验中的证据函数$f$，即达到最大均差的函数，总是一个二次样条函数。

    Maximum mean discrepancy (MMD) refers to a general class of nonparametric two-sample tests that are based on maximizing the mean difference over samples from one distribution $P$ versus another $Q$, over all choices of data transformations $f$ living in some function space $\mathcal{F}$. Inspired by recent work that connects what are known as functions of $\textit{Radon bounded variation}$ (RBV) and neural networks (Parhi and Nowak, 2021, 2023), we study the MMD defined by taking $\mathcal{F}$ to be the unit ball in the RBV space of a given smoothness order $k \geq 0$. This test, which we refer to as the $\textit{Radon-Kolmogorov-Smirnov}$ (RKS) test, can be viewed as a generalization of the well-known and classical Kolmogorov-Smirnov (KS) test to multiple dimensions and higher orders of smoothness. It is also intimately connected to neural networks: we prove that the witness in the RKS test -- the function $f$ achieving the maximum mean difference -- is always a ridge spline of degree
    

