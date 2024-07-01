# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Networked Communication for Decentralised Agents in Mean-Field Games.](http://arxiv.org/abs/2306.02766) | 本研究在均场博弈中引入网络通信，提出了一种提高分布式智能体学习效率的方案，并进行了实际实验验证。 |
| [^2] | [The G-invariant graph Laplacian.](http://arxiv.org/abs/2303.17001) | 本文提出了 G不变图拉普拉斯算子 用于处理数据集不仅在流形上，而且在一个连续群的作用下也是封闭的情形，相较于标准图拉普拉斯算子收敛速度更快。 |

# 详细

[^1]: 分布式智能体在均场博弈中的网络通信

    Networked Communication for Decentralised Agents in Mean-Field Games. (arXiv:2306.02766v2 [cs.MA] UPDATED)

    [http://arxiv.org/abs/2306.02766](http://arxiv.org/abs/2306.02766)

    本研究在均场博弈中引入网络通信，提出了一种提高分布式智能体学习效率的方案，并进行了实际实验验证。

    

    我们将网络通信引入均场博弈框架，特别是在无oracle的情况下，N个分布式智能体沿着经过的经验系统的单一非周期演化路径学习。我们证明，我们的架构在只有一些关于网络结构的合理假设的情况下，具有样本保证，在集中学习和独立学习情况之间有界。我们讨论了三个理论算法的样本保证实际上并不会导致实际收敛。因此，我们展示了在实际设置中，当理论参数未被观察到（导致Q函数的估计不准确）时，我们的通信方案显著加速了收敛速度，而无需依赖于一个不可取的集中式控制器的假设。我们对三个理论算法进行了几种实际的改进，使我们能够展示它们的第一个实证表现。

    We introduce networked communication to the mean-field game framework, in particular to oracle-free settings where $N$ decentralised agents learn along a single, non-episodic evolution path of the empirical system. We prove that our architecture, with only a few reasonable assumptions about network structure, has sample guarantees bounded between those of the centralised- and independent-learning cases. We discuss how the sample guarantees of the three theoretical algorithms do not actually result in practical convergence. Accordingly, we show that in practical settings where the theoretical parameters are not observed (leading to poor estimation of the Q-function), our communication scheme significantly accelerates convergence over the independent case, without relying on the undesirable assumption of a centralised controller. We contribute several further practical enhancements to all three theoretical algorithms, allowing us to showcase their first empirical demonstrations. Our expe
    
[^2]: G不变图拉普拉斯算子

    The G-invariant graph Laplacian. (arXiv:2303.17001v1 [cs.LG])

    [http://arxiv.org/abs/2303.17001](http://arxiv.org/abs/2303.17001)

    本文提出了 G不变图拉普拉斯算子 用于处理数据集不仅在流形上，而且在一个连续群的作用下也是封闭的情形，相较于标准图拉普拉斯算子收敛速度更快。

    

    基于图拉普拉斯算子的算法已经被证明在降维、聚类和去噪等领域对流形数据非常有效。本文考虑的数据集不仅在流形上，而且在一个连续群的作用下也是封闭的。这类数据集的一个例子是沿着低维流形传播的体积，其中每个体积可以在三维空间中旋转。我们介绍了G不变图拉普拉斯算子，通过考虑数据集上的群的作用来广义化图拉普拉斯算子。我们显示了与标准图拉普拉斯算子类似，G不变图拉普拉斯算子收敛于数据流形上的Laplace-Beltrami算子，但收敛速度显著提高。此外，我们还表明G不变图拉普拉斯算子的特征函数具有群元素和某些矩阵的特征向量的张量积形式，可以使用F高效地计算。

    Graph Laplacian based algorithms for data lying on a manifold have been proven effective for tasks such as dimensionality reduction, clustering, and denoising. In this work, we consider data sets whose data point not only lie on a manifold, but are also closed under the action of a continuous group. An example of such data set is volumes that line on a low dimensional manifold, where each volume may be rotated in three-dimensional space. We introduce the G-invariant graph Laplacian that generalizes the graph Laplacian by accounting for the action of the group on the data set. We show that like the standard graph Laplacian, the G-invariant graph Laplacian converges to the Laplace-Beltrami operator on the data manifold, but with a significantly improved convergence rate. Furthermore, we show that the eigenfunctions of the G-invariant graph Laplacian admit the form of tensor products between the group elements and eigenvectors of certain matrices, which can be computed efficiently using F
    

