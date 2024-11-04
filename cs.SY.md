# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PowerGraph: A power grid benchmark dataset for graph neural networks](https://arxiv.org/abs/2402.02827) | PowerGraph是一个用于图神经网络的电网基准数据集，旨在通过机器学习模型实现电力网格断电的在线检测。 |
| [^2] | [Decentralized Online Regularized Learning Over Random Time-Varying Graphs.](http://arxiv.org/abs/2206.03861) | 本文研究了随机时变图上的分散在线正则化线性回归算法，提出了非负超-鞅不等式的估计误差，证明了算法在满足样本路径时空兴奋条件时，节点的估计可以收敛于未知的真实参数向量。 |

# 详细

[^1]: PowerGraph: 用于图神经网络的电网基准数据集

    PowerGraph: A power grid benchmark dataset for graph neural networks

    [https://arxiv.org/abs/2402.02827](https://arxiv.org/abs/2402.02827)

    PowerGraph是一个用于图神经网络的电网基准数据集，旨在通过机器学习模型实现电力网格断电的在线检测。

    

    公共图神经网络（GNN）基准数据集有助于使用GNN，并增强GNN在各个领域中的适用性。目前，社区中缺乏用于GNN应用的电力网格公共数据集。事实上，与其他机器学习技术相比，GNN可以潜在地捕捉到复杂的电力网格现象。电力网格是复杂的工程网络，天然适合于图表示。因此，GNN有潜力捕捉到电力网格的行为，而不用其他机器学习技术。为了实现这个目标，我们开发了一个用于级联故障事件的图数据集，这是导致电力网格断电的主要原因。历史断电数据集稀缺且不完整。通常通过计算昂贵的离线级联故障模拟来评估脆弱性和识别关键组件。相反，我们建议使用机器学习模型进行在线检测。

    Public Graph Neural Networks (GNN) benchmark datasets facilitate the use of GNN and enhance GNN applicability to diverse disciplines. The community currently lacks public datasets of electrical power grids for GNN applications. Indeed, GNNs can potentially capture complex power grid phenomena over alternative machine learning techniques. Power grids are complex engineered networks that are naturally amenable to graph representations. Therefore, GNN have the potential for capturing the behavior of power grids over alternative machine learning techniques. To this aim, we develop a graph dataset for cascading failure events, which are the major cause of blackouts in electric power grids. Historical blackout datasets are scarce and incomplete. The assessment of vulnerability and the identification of critical components are usually conducted via computationally expensive offline simulations of cascading failures. Instead, we propose using machine learning models for the online detection of
    
[^2]: 随机时变图上的分散在线正则化学习

    Decentralized Online Regularized Learning Over Random Time-Varying Graphs. (arXiv:2206.03861v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.03861](http://arxiv.org/abs/2206.03861)

    本文研究了随机时变图上的分散在线正则化线性回归算法，提出了非负超-鞅不等式的估计误差，证明了算法在满足样本路径时空兴奋条件时，节点的估计可以收敛于未知的真实参数向量。

    

    本文研究了在随机时变图上的分散在线正则化线性回归算法。在每个时间步中，每个节点都运行一个在线估计算法，该算法包括创新项（处理自身新测量值）、共识项（加权平均自身及其邻居的估计，带有加性和乘性通信噪声）和正则化项（防止过度拟合）。不要求回归矩阵和图满足特殊的统计假设，如相互独立、时空独立或平稳性。我们发展了非负超-鞅不等式的估计误差，并证明了如果算法增益、图和回归矩阵共同满足样本路径时空兴奋条件，节点的估计几乎可以肯定地收敛于未知的真实参数向量。特别地，通过选择适当的算法增益，该条件成立。

    We study the decentralized online regularized linear regression algorithm over random time-varying graphs. At each time step, every node runs an online estimation algorithm consisting of an innovation term processing its own new measurement, a consensus term taking a weighted sum of estimations of its own and its neighbors with additive and multiplicative communication noises and a regularization term preventing over-fitting. It is not required that the regression matrices and graphs satisfy special statistical assumptions such as mutual independence, spatio-temporal independence or stationarity. We develop the nonnegative supermartingale inequality of the estimation error, and prove that the estimations of all nodes converge to the unknown true parameter vector almost surely if the algorithm gains, graphs and regression matrices jointly satisfy the sample path spatio-temporal persistence of excitation condition. Especially, this condition holds by choosing appropriate algorithm gains 
    

