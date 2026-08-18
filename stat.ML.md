# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ODTlearn: A Package for Learning Optimal Decision Trees for Prediction and Prescription.](http://arxiv.org/abs/2307.15691) | ODTlearn是一个开源的Python包，用于学习预测和处方的最优决策树。它提供了多种优化方法，并支持各种问题和算法的扩展。 |
| [^2] | [Spectral clustering in the Gaussian mixture block model.](http://arxiv.org/abs/2305.00979) | 本文首次研究了从高维高斯混合块模型中抽样的图聚类和嵌入问题。 |
| [^3] | [Doubly robust nearest neighbors in factor models.](http://arxiv.org/abs/2211.14297) | 该论文介绍了一种在潜在因子模型中处理缺失数据的双重稳健最近邻方法，可以提供一致的估计，并在存在良好的行和列邻居时提供（近似）二次改进非渐近性能。 |

# 详细

[^1]: ODTlearn: 一个用于学习预测和处方的最优决策树的包

    ODTlearn: A Package for Learning Optimal Decision Trees for Prediction and Prescription. (arXiv:2307.15691v1 [stat.ML])

    [http://arxiv.org/abs/2307.15691](http://arxiv.org/abs/2307.15691)

    ODTlearn是一个开源的Python包，用于学习预测和处方的最优决策树。它提供了多种优化方法，并支持各种问题和算法的扩展。

    

    ODTLearn是一个开源的Python包，提供了基于混合整数优化(MIO)框架的高风险预测和处方任务的最优决策树学习方法。该包的当前版本提供了学习最优分类树、公平最优分类树、鲁棒最优分类树和从观测数据学习最优处方树的实现。我们设计了该包以便于维护和扩展，当引入新的最优决策树问题类、重构策略和解决算法时，可以轻松更新。为此，该包遵循面向对象的设计原则，并支持商业(Gurobi)和开源(COIN-OR branch and cut)求解器。包的文档和详细用户指南可以在https://d3m-research-group.github.io/odtlearn/找到。

    ODTLearn is an open-source Python package that provides methods for learning optimal decision trees for high-stakes predictive and prescriptive tasks based on the mixed-integer optimization (MIO) framework proposed in Aghaei et al. (2019) and several of its extensions. The current version of the package provides implementations for learning optimal classification trees, optimal fair classification trees, optimal classification trees robust to distribution shifts, and optimal prescriptive trees from observational data. We have designed the package to be easy to maintain and extend as new optimal decision tree problem classes, reformulation strategies, and solution algorithms are introduced. To this end, the package follows object-oriented design principles and supports both commercial (Gurobi) and open source (COIN-OR branch and cut) solvers. The package documentation and an extensive user guide can be found at https://d3m-research-group.github.io/odtlearn/. Additionally, users can view
    
[^2]: 高斯混合块模型中的谱聚类

    Spectral clustering in the Gaussian mixture block model. (arXiv:2305.00979v1 [stat.ML])

    [http://arxiv.org/abs/2305.00979](http://arxiv.org/abs/2305.00979)

    本文首次研究了从高维高斯混合块模型中抽样的图聚类和嵌入问题。

    

    高斯混合块模型是用于模拟现代网络的图分布：对于这样的模型生成一个图，我们将每个顶点 $i$ 与一个从高斯混合中抽样到的潜在特征向量 $u_i \in \mathbb{R}^d$ 相关联，当且仅当特征向量足够相似，即 $\langle u_i,u_j \rangle \ge \tau$ 时，我们才会添加边 $(i,j)$。高斯混合的不同组成部分表示可能具有不同特征分布的不同类型的节点，例如在社交网络中，每个组成部分都表示独特社区的不同属性。这些网络涉及到的自然算法任务有嵌入（恢复潜在的特征向量）和聚类（通过其混合组分将节点分组）。本文开启了对从高维高斯混合块模型抽样的图进行聚类和嵌入研究。

    Gaussian mixture block models are distributions over graphs that strive to model modern networks: to generate a graph from such a model, we associate each vertex $i$ with a latent feature vector $u_i \in \mathbb{R}^d$ sampled from a mixture of Gaussians, and we add edge $(i,j)$ if and only if the feature vectors are sufficiently similar, in that $\langle u_i,u_j \rangle \ge \tau$ for a pre-specified threshold $\tau$. The different components of the Gaussian mixture represent the fact that there may be different types of nodes with different distributions over features -- for example, in a social network each component represents the different attributes of a distinct community. Natural algorithmic tasks associated with these networks are embedding (recovering the latent feature vectors) and clustering (grouping nodes by their mixture component).  In this paper we initiate the study of clustering and embedding graphs sampled from high-dimensional Gaussian mixture block models, where the
    
[^3]: 因子模型中的双重稳健最近邻方法

    Doubly robust nearest neighbors in factor models. (arXiv:2211.14297v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2211.14297](http://arxiv.org/abs/2211.14297)

    该论文介绍了一种在潜在因子模型中处理缺失数据的双重稳健最近邻方法，可以提供一致的估计，并在存在良好的行和列邻居时提供（近似）二次改进非渐近性能。

    

    我们介绍并分析了在潜在因子模型中处理缺失数据的改进最近邻（NN）方法。我们考虑一个带有缺失数据的矩阵补全问题，其中当被观察到时，第$(i, t)$个条目由其均值$f(u_i, v_t)$加上均值为零的噪声给出，其中$f$为未知函数，$u_i$和$v_t$为潜在因子。之前的NN策略，如单元-单元NN，用于估计均值$f(u_i, v_t)$，依赖于存在其他行$j$使得$u_j \approx u_i$。类似地，时间-时间NN策略依赖于存在列$t'$使得$v_{t'} \approx v_t$。当相似行或相似列不可用时，这些策略的性能较差。我们的估计在两个方面对这种不足是双重稳健的：(1) 只要存在良好的行或列邻居，我们的估计提供一致的估计。 (2) 此外，如果存在良好的行和列邻居，它提供了（近似）二次改进非渐近性能。

    We introduce and analyze an improved variant of nearest neighbors (NN) for estimation with missing data in latent factor models. We consider a matrix completion problem with missing data, where the $(i, t)$-th entry, when observed, is given by its mean $f(u_i, v_t)$ plus mean-zero noise for an unknown function $f$ and latent factors $u_i$ and $v_t$. Prior NN strategies, like unit-unit NN, for estimating the mean $f(u_i, v_t)$ relies on existence of other rows $j$ with $u_j \approx u_i$. Similarly, time-time NN strategy relies on existence of columns $t'$ with $v_{t'} \approx v_t$. These strategies provide poor performance respectively when similar rows or similar columns are not available. Our estimate is doubly robust to this deficit in two ways: (1) As long as there exist either good row or good column neighbors, our estimate provides a consistent estimate. (2) Furthermore, if both good row and good column neighbors exist, it provides a (near-)quadratic improvement in the non-asympto
    

