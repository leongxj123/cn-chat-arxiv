# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Active Adaptive Experimental Design for Treatment Effect Estimation with Covariate Choices](https://arxiv.org/abs/2403.03589) | 该研究提出了一种更有效地估计处理效应的活跃自适应实验设计方法，通过优化协变量密度和倾向得分来降低渐近方差。 |
| [^2] | [On the use of the Gram matrix for multivariate functional principal components analysis.](http://arxiv.org/abs/2306.12949) | 本文提出使用内积来估计多元和多维函数数据集的特征向量，为函数主成分分析提供了新的有效方法。 |
| [^3] | [Imputation of missing values in multi-view data.](http://arxiv.org/abs/2210.14484) | 本文提出了一种基于StaPLR算法的新的多视角数据插补算法，通过在降维空间中执行插补以解决计算挑战，并在模拟数据集中得到了竞争性结果。 |

# 详细

[^1]: 用于处理因变量选择的活跃自适应实验设计的处理效应估计

    Active Adaptive Experimental Design for Treatment Effect Estimation with Covariate Choices

    [https://arxiv.org/abs/2403.03589](https://arxiv.org/abs/2403.03589)

    该研究提出了一种更有效地估计处理效应的活跃自适应实验设计方法，通过优化协变量密度和倾向得分来降低渐近方差。

    

    这项研究设计了一个自适应实验，用于高效地估计平均处理效应（ATEs）。我们考虑了一个自适应实验，其中实验者按顺序从由实验者决定的协变量密度中抽样一个实验单元，并分配一种处理。在分配处理后，实验者立即观察相应的结果。在实验结束时，实验者利用收集的样本估算出一个ATE。实验者的目标是通过较小的渐近方差估计ATE。现有研究已经设计了一些能够自适应优化倾向得分（处理分配概率）的实验。作为这种方法的一个概括，我们提出了一个框架，该框架下实验者优化协变量密度以及倾向得分，并发现优化协变量密度和倾向得分比仅优化倾向得分可以减少渐近方差更多的情况。

    arXiv:2403.03589v1 Announce Type: cross  Abstract: This study designs an adaptive experiment for efficiently estimating average treatment effect (ATEs). We consider an adaptive experiment where an experimenter sequentially samples an experimental unit from a covariate density decided by the experimenter and assigns a treatment. After assigning a treatment, the experimenter observes the corresponding outcome immediately. At the end of the experiment, the experimenter estimates an ATE using gathered samples. The objective of the experimenter is to estimate the ATE with a smaller asymptotic variance. Existing studies have designed experiments that adaptively optimize the propensity score (treatment-assignment probability). As a generalization of such an approach, we propose a framework under which an experimenter optimizes the covariate density, as well as the propensity score, and find that optimizing both covariate density and propensity score reduces the asymptotic variance more than o
    
[^2]: 关于使用格拉姆矩阵进行多元函数主成分分析的研究

    On the use of the Gram matrix for multivariate functional principal components analysis. (arXiv:2306.12949v1 [stat.ME])

    [http://arxiv.org/abs/2306.12949](http://arxiv.org/abs/2306.12949)

    本文提出使用内积来估计多元和多维函数数据集的特征向量，为函数主成分分析提供了新的有效方法。

    

    在函数数据分析中，降维是至关重要的。降维的关键工具是函数主成分分析。现有的函数主成分分析方法通常涉及协方差矩阵的对角化。随着函数数据集的规模和复杂性增加，协方差矩阵的估计变得更加具有挑战性。因此，需要有效的方法来估计特征向量。基于观测空间和函数特征空间的对偶性，我们提出使用曲线之间的内积来估计多元和多维函数数据集的特征向量。建立了协方差矩阵特征向量和内积矩阵特征向量之间的关系。我们探讨了这些方法在几个函数数据分析设置中的应用，并提供了它们的通用指导。

    Dimension reduction is crucial in functional data analysis (FDA). The key tool to reduce the dimension of the data is functional principal component analysis. Existing approaches for functional principal component analysis usually involve the diagonalization of the covariance operator. With the increasing size and complexity of functional datasets, estimating the covariance operator has become more challenging. Therefore, there is a growing need for efficient methodologies to estimate the eigencomponents. Using the duality of the space of observations and the space of functional features, we propose to use the inner-product between the curves to estimate the eigenelements of multivariate and multidimensional functional datasets. The relationship between the eigenelements of the covariance operator and those of the inner-product matrix is established. We explore the application of these methodologies in several FDA settings and provide general guidance on their usability.
    
[^3]: 多视角数据中缺失值的插补问题解决方法

    Imputation of missing values in multi-view data. (arXiv:2210.14484v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2210.14484](http://arxiv.org/abs/2210.14484)

    本文提出了一种基于StaPLR算法的新的多视角数据插补算法，通过在降维空间中执行插补以解决计算挑战，并在模拟数据集中得到了竞争性结果。

    

    多视角数据是指由多个不同特征集描述的数据。在处理多视角数据时，若出现缺失值，则一个视角中的所有特征极有可能同时缺失，因而导致非常大量的缺失数据问题。本文提出了一种新的多视角学习算法中的插补方法，它基于堆叠惩罚逻辑回归(StaPLR)算法，在降维空间中执行插补，以解决固有的多视角计算挑战。实验结果表明，该方法在模拟数据集上具有竞争性结果，而且具有更低的计算成本，从而可以使用先进的插补算法，例如missForest。

    Data for which a set of objects is described by multiple distinct feature sets (called views) is known as multi-view data. When missing values occur in multi-view data, all features in a view are likely to be missing simultaneously. This leads to very large quantities of missing data which, especially when combined with high-dimensionality, makes the application of conditional imputation methods computationally infeasible. We introduce a new imputation method based on the existing stacked penalized logistic regression (StaPLR) algorithm for multi-view learning. It performs imputation in a dimension-reduced space to address computational challenges inherent to the multi-view context. We compare the performance of the new imputation method with several existing imputation algorithms in simulated data sets. The results show that the new imputation method leads to competitive results at a much lower computational cost, and makes the use of advanced imputation algorithms such as missForest 
    

