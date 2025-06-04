# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning WENO for entropy stable schemes to solve conservation laws](https://arxiv.org/abs/2403.14848) | 提出了一种称为Deep Sign-Preserving WENO（DSP-WENO）的变种，通过神经网络学习WENO加权策略，以改进在震荡附近表现不佳的WENO算法。 |
| [^2] | [Shallow ReLU neural networks and finite elements](https://arxiv.org/abs/2403.05809) | 在凸多面体网格上，提出了用两个隐藏层的ReLU神经网络来弱表示分段线性函数，并根据网格中的多面体和超平面的数量准确确定了所需的神经元数，建立了浅层ReLU神经网络和有限元函数之间的联系。 |
| [^3] | [Why Shallow Networks Struggle with Approximating and Learning High Frequency: A Numerical Study.](http://arxiv.org/abs/2306.17301) | 本文通过数值研究探讨了浅层神经网络在逼近和学习高频率方面的困难，重点是通过分析激活函数的谱分析来理解问题的原因。 |

# 详细

[^1]: 学习WENO用于熵稳定方案以解决守恒定律

    Learning WENO for entropy stable schemes to solve conservation laws

    [https://arxiv.org/abs/2403.14848](https://arxiv.org/abs/2403.14848)

    提出了一种称为Deep Sign-Preserving WENO（DSP-WENO）的变种，通过神经网络学习WENO加权策略，以改进在震荡附近表现不佳的WENO算法。

    

    熵条件在提取系统守恒律的物理相关解时起着至关重要的作用，因此促使构建满足离散条件的熵稳定方案。 TeCNO方案（Fjordholm等，2012）形成了一类任意高阶熵稳定有限差分求解器，它们需要满足每个单元格界面的符号特性的专业重构算法。最近，设计了满足符号特性的第三阶WENO方案，称为SP-WENO（Fjordholm和Ray，2016）和SP-WENOc（Ray，2018）。然而，这些WENO算法在震荡附近的性能可能很差，数值解表现出大的人工振荡。在本研究中，我们提出了SP-WENO的一个变种，称为Deep Sign-Preserving WENO（DSP-WENO），在其中，一个神经网络被训练来学习WENO加权策略。

    arXiv:2403.14848v1 Announce Type: cross  Abstract: Entropy conditions play a crucial role in the extraction of a physically relevant solution for a system of conservation laws, thus motivating the construction of entropy stable schemes that satisfy a discrete analogue of such conditions. TeCNO schemes (Fjordholm et al. 2012) form a class of arbitrary high-order entropy stable finite difference solvers, which require specialized reconstruction algorithms satisfying the sign property at each cell interface. Recently, third-order WENO schemes called SP-WENO (Fjordholm and Ray, 2016) and SP-WENOc (Ray, 2018) have been designed to satisfy the sign property. However, these WENO algorithms can perform poorly near shocks, with the numerical solutions exhibiting large spurious oscillations. In the present work, we propose a variant of the SP-WENO, termed as Deep Sign-Preserving WENO (DSP-WENO), where a neural network is trained to learn the WENO weighting strategy. The sign property and third-o
    
[^2]: 浅层ReLU神经网络和有限元

    Shallow ReLU neural networks and finite elements

    [https://arxiv.org/abs/2403.05809](https://arxiv.org/abs/2403.05809)

    在凸多面体网格上，提出了用两个隐藏层的ReLU神经网络来弱表示分段线性函数，并根据网格中的多面体和超平面的数量准确确定了所需的神经元数，建立了浅层ReLU神经网络和有限元函数之间的联系。

    

    我们指出在凸多面体网格上，可以用两个隐藏层的ReLU神经网络在弱意义下表示（连续或不连续的）分段线性函数。此外，基于涉及到的多面体和超平面的数量，准确给出了弱表示所需的两个隐藏层的神经元数。这些结果自然地适用于常数和线性有限元函数。这种弱表示建立了浅层ReLU神经网络和有限元函数之间的桥梁，并为通过有限元函数分析ReLU神经网络在$L^p$范数中的逼近能力提供了视角。此外，我们还讨论了最近张量神经网络对张量有限元函数的严格表示。

    arXiv:2403.05809v1 Announce Type: cross  Abstract: We point out that (continuous or discontinuous) piecewise linear functions on a convex polytope mesh can be represented by two-hidden-layer ReLU neural networks in a weak sense. In addition, the numbers of neurons of the two hidden layers required to weakly represent are accurately given based on the numbers of polytopes and hyperplanes involved in this mesh. The results naturally hold for constant and linear finite element functions. Such weak representation establishes a bridge between shallow ReLU neural networks and finite element functions, and leads to a perspective for analyzing approximation capability of ReLU neural networks in $L^p$ norm via finite element functions. Moreover, we discuss the strict representation for tensor finite element functions via the recent tensor neural networks.
    
[^3]: 浅层网络在逼近和学习高频率方面的困难：一个数值研究

    Why Shallow Networks Struggle with Approximating and Learning High Frequency: A Numerical Study. (arXiv:2306.17301v1 [cs.LG])

    [http://arxiv.org/abs/2306.17301](http://arxiv.org/abs/2306.17301)

    本文通过数值研究探讨了浅层神经网络在逼近和学习高频率方面的困难，重点是通过分析激活函数的谱分析来理解问题的原因。

    

    本研究通过对分析和实验的综合数值研究，解释了为什么两层神经网络在机器精度和计算成本等实际因素中，处理高频率的逼近和学习存在困难。具体而言，研究了以下基本计算问题：（1）在有限的机器精度下可以达到的最佳精度，（2）实现给定精度所需的计算成本，以及（3）对扰动的稳定性。研究的关键是相应激活函数的格拉姆矩阵的谱分析，该分析还显示了激活函数属性在这个问题中的作用。

    In this work, a comprehensive numerical study involving analysis and experiments shows why a two-layer neural network has difficulties handling high frequencies in approximation and learning when machine precision and computation cost are important factors in real practice. In particular, the following fundamental computational issues are investigated: (1) the best accuracy one can achieve given a finite machine precision, (2) the computation cost to achieve a given accuracy, and (3) stability with respect to perturbations. The key to the study is the spectral analysis of the corresponding Gram matrix of the activation functions which also shows how the properties of the activation function play a role in the picture.
    

