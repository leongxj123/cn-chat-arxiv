# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A model for efficient dynamical ranking in networks.](http://arxiv.org/abs/2307.13544) | 该论文提出了一种受物理启发的方法，用于在定向时态网络中推断节点的动态排序，通过求解线性方程组实现，仅需调整一个参数，具有可扩展性和高效性。在各种应用中的测试结果显示，该方法比现有方法更好地预测了动态排序。 |

# 详细

[^1]: 在网络中高效动态排序的模型

    A model for efficient dynamical ranking in networks. (arXiv:2307.13544v1 [physics.soc-ph])

    [http://arxiv.org/abs/2307.13544](http://arxiv.org/abs/2307.13544)

    该论文提出了一种受物理启发的方法，用于在定向时态网络中推断节点的动态排序，通过求解线性方程组实现，仅需调整一个参数，具有可扩展性和高效性。在各种应用中的测试结果显示，该方法比现有方法更好地预测了动态排序。

    

    我们提出了一种受物理启发的方法，用于推断定向时态网络中的动态排序 - 每个定向且带时间戳的边反映了一对交互的结果和时间。每个节点的推断排序是实值且随时间变化的，每次新的边缘都会提高或降低节点的估计强度或声望，这在真实情景中经常观察到，包括游戏序列，锦标赛或动物等级的相互作用。我们的方法通过求解一组线性方程来工作，并且只需要调整一个参数。因此，对应的算法是可扩展且高效的。我们通过评估方法在各种应用中预测交互（边缘存在）及其结果（边缘方向）的能力来测试我们的方法，包括合成数据和真实数据。我们的分析显示，在许多情况下，我们的方法比现有的方法更好地预测了动态排序。

    We present a physics-inspired method for inferring dynamic rankings in directed temporal networks - networks in which each directed and timestamped edge reflects the outcome and timing of a pairwise interaction. The inferred ranking of each node is real-valued and varies in time as each new edge, encoding an outcome like a win or loss, raises or lowers the node's estimated strength or prestige, as is often observed in real scenarios including sequences of games, tournaments, or interactions in animal hierarchies. Our method works by solving a linear system of equations and requires only one parameter to be tuned. As a result, the corresponding algorithm is scalable and efficient. We test our method by evaluating its ability to predict interactions (edges' existence) and their outcomes (edges' directions) in a variety of applications, including both synthetic and real data. Our analysis shows that in many cases our method's performance is better than existing methods for predicting dyna
    

