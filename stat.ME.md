# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Bayesian Context Trees State Space Model for time series modelling and forecasting.](http://arxiv.org/abs/2308.00913) | 该论文介绍了基于贝叶斯上下文树状态空间模型的时间序列建模和预测方法，通过层级贝叶斯框架将离散状态和实值时间序列模型组合，构建出灵活且可解释的混合模型，并提出了有效的算法来进行贝叶斯推断和预测。 |
| [^2] | [High-Dimensional Bayesian Structure Learning in Gaussian Graphical Models using Marginal Pseudo-Likelihood.](http://arxiv.org/abs/2307.00127) | 该论文提出了两种创新的搜索算法，在高维图结构学习中使用边际伪似然函数解决计算复杂性问题，并且能够在短时间内生成可靠的估计。该方法提供了R软件包BDgraph的代码实现。 |

# 详细

[^1]: 基于贝叶斯上下文树状态空间模型的时间序列建模和预测

    The Bayesian Context Trees State Space Model for time series modelling and forecasting. (arXiv:2308.00913v1 [stat.ME])

    [http://arxiv.org/abs/2308.00913](http://arxiv.org/abs/2308.00913)

    该论文介绍了基于贝叶斯上下文树状态空间模型的时间序列建模和预测方法，通过层级贝叶斯框架将离散状态和实值时间序列模型组合，构建出灵活且可解释的混合模型，并提出了有效的算法来进行贝叶斯推断和预测。

    

    引入了一个层级贝叶斯框架，用于开发用于真实值时间序列的丰富混合模型，以及一系列有效的学习和推断工具。在顶层，通过适当量化最近样本的一些有意义的离散状态来进行鉴定。这些可观察状态的集合被描述为离散的上下文树模型。然后，在底层，将一个不同的、任意的实值时间序列模型（基本模型）与每个状态相关联。这定义了一个非常通用的框架，可以与任何现有模型类一起使用，构建灵活且可解释的混合模型。我们将其称为贝叶斯上下文树状态空间模型，或者BCT-X框架。引入了高效的算法，可以实现有效的、精确的贝叶斯推断；特别是可以确定最大后验概率（MAP）上下文树模型。这些算法可以顺序更新，以便实现有效的推断和预测。

    A hierarchical Bayesian framework is introduced for developing rich mixture models for real-valued time series, along with a collection of effective tools for learning and inference. At the top level, meaningful discrete states are identified as appropriately quantised values of some of the most recent samples. This collection of observable states is described as a discrete context-tree model. Then, at the bottom level, a different, arbitrary model for real-valued time series - a base model - is associated with each state. This defines a very general framework that can be used in conjunction with any existing model class to build flexible and interpretable mixture models. We call this the Bayesian Context Trees State Space Model, or the BCT-X framework. Efficient algorithms are introduced that allow for effective, exact Bayesian inference; in particular, the maximum a posteriori probability (MAP) context-tree model can be identified. These algorithms can be updated sequentially, facili
    
[^2]: 高维贝叶斯高斯图模型中的结构学习方法——利用边际伪似然函数

    High-Dimensional Bayesian Structure Learning in Gaussian Graphical Models using Marginal Pseudo-Likelihood. (arXiv:2307.00127v1 [stat.ME])

    [http://arxiv.org/abs/2307.00127](http://arxiv.org/abs/2307.00127)

    该论文提出了两种创新的搜索算法，在高维图结构学习中使用边际伪似然函数解决计算复杂性问题，并且能够在短时间内生成可靠的估计。该方法提供了R软件包BDgraph的代码实现。

    

    高斯图模型以图形形式描绘了多元正态分布中变量之间的条件依赖关系。这篇论文介绍了两种创新的搜索算法，利用边际伪似然函数来应对高维图结构学习中的计算复杂性问题。这些方法可以在标准计算机上在几分钟内快速生成对包含1000个变量的问题的可靠估计。对于对实际应用感兴趣的人，支持这种新方法的代码通过R软件包BDgraph提供。

    Gaussian graphical models depict the conditional dependencies between variables within a multivariate normal distribution in a graphical format. The identification of these graph structures is an area known as structure learning. However, when utilizing Bayesian methodologies in structure learning, computational complexities can arise, especially with high-dimensional graphs surpassing 250 nodes. This paper introduces two innovative search algorithms that employ marginal pseudo-likelihood to address this computational challenge. These methods can swiftly generate reliable estimations for problems encompassing 1000 variables in just a few minutes on standard computers. For those interested in practical applications, the code supporting this new approach is made available through the R package BDgraph.
    

