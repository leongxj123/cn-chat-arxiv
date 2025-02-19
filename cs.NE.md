# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Calibration Error Estimation Using Fuzzy Binning.](http://arxiv.org/abs/2305.00543) | 本文提出了一种模糊校准误差度量（FCE），利用模糊分箱方法计算校准误差，从而缓解了概率偏斜的影响并提供了更紧密的估计值。与传统指标ECE相比，FCE在多类设置中表现更好，https://github.com/srdgFHE/FCE-paper。 |
| [^2] | [Granular-ball Optimization Algorithm.](http://arxiv.org/abs/2303.12807) | 粒球优化算法(GBO)是一种新的多粒度优化算法，可以通过引入粒球计算来提高全局搜索能力和收敛速度，实验结果表明，在这些方面它比现有的最先进的算法表现更优。 |

# 详细

[^1]: 使用模糊分箱进行校准误差估计

    Calibration Error Estimation Using Fuzzy Binning. (arXiv:2305.00543v1 [cs.LG])

    [http://arxiv.org/abs/2305.00543](http://arxiv.org/abs/2305.00543)

    本文提出了一种模糊校准误差度量（FCE），利用模糊分箱方法计算校准误差，从而缓解了概率偏斜的影响并提供了更紧密的估计值。与传统指标ECE相比，FCE在多类设置中表现更好，https://github.com/srdgFHE/FCE-paper。

    

    基于神经网络的决策往往会过于自信，其原始结果的概率并不符合真实的决策概率。神经网络的校准是实现更可靠的深度学习框架的关键步骤。先前的校准误差度量主要利用清晰的分箱成员资格度量。这加剧了模型概率的偏斜，并描绘了校准误差的不完整图像。在本文中，我们提出了一种利用模糊分箱方法计算校准误差的模糊校准误差度量（FCE）。这种方法缓解了概率偏斜的影响，并在测量校准误差时提供了更紧密的估计值。我们比较了我们的指标与ECE在不同的数据群体和类别成员身份中的表现。我们的结果显示，FCE在校准误差估计方面表现更好，特别是在多类设置中，缓解了模型置信度分数偏斜对校准误差估计的影响。我们提供了我们的代码https://github.com/srdgFHE/FCE-paper，以便未来的可重复性和使用FCE进行校准误差估计。

    Neural network-based decisions tend to be overconfident, where their raw outcome probabilities do not align with the true decision probabilities. Calibration of neural networks is an essential step towards more reliable deep learning frameworks. Prior metrics of calibration error primarily utilize crisp bin membership-based measures. This exacerbates skew in model probabilities and portrays an incomplete picture of calibration error. In this work, we propose a Fuzzy Calibration Error metric (FCE) that utilizes a fuzzy binning approach to calculate calibration error. This approach alleviates the impact of probability skew and provides a tighter estimate while measuring calibration error. We compare our metric with ECE across different data populations and class memberships. Our results show that FCE offers better calibration error estimation, especially in multi-class settings, alleviating the effects of skew in model confidence scores on calibration error estimation. We make our code a
    
[^2]: 粒球优化算法

    Granular-ball Optimization Algorithm. (arXiv:2303.12807v1 [cs.LG])

    [http://arxiv.org/abs/2303.12807](http://arxiv.org/abs/2303.12807)

    粒球优化算法(GBO)是一种新的多粒度优化算法，可以通过引入粒球计算来提高全局搜索能力和收敛速度，实验结果表明，在这些方面它比现有的最先进的算法表现更优。

    

    现有的智能优化算法都是基于最小粒度即点的设计，导致全局搜索能力较弱且效率低下。为了解决这个问题，我们提出了一种新的多粒度优化算法，即粒球优化算法(GBO)，通过引入粒球计算来实现。GBO使用多个粒球来覆盖解空间，使用许多细小的细粒度粒球来描述重要部分，使用少量的大粗粒度粒球来描述不重要的部分，精细的多粒度数据描述能力提高了全局搜索能力和收敛速度。针对二十个基准函数的实验结果表明，与最流行的最先进的算法相比，GBO具有更好的性能和更快的速度，更接近最优解，没有超参数，设计更简单。

    The existing intelligent optimization algorithms are designed based on the finest granularity, i.e., a point. This leads to weak global search ability and inefficiency. To address this problem, we proposed a novel multi-granularity optimization algorithm, namely granular-ball optimization algorithm (GBO), by introducing granular-ball computing. GBO uses many granular-balls to cover the solution space. Quite a lot of small and fine-grained granular-balls are used to depict the important parts, and a little number of large and coarse-grained granular-balls are used to depict the inessential parts. Fine multi-granularity data description ability results in a higher global search capability and faster convergence speed. In comparison with the most popular and state-of-the-art algorithms, the experiments on twenty benchmark functions demonstrate its better performance. The faster speed, higher approximation ability of optimal solution, no hyper-parameters, and simpler design of GBO make it 
    

