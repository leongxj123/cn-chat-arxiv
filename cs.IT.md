# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generalized Lagrange Coded Computing: A Flexible Computation-Communication Tradeoff for Resilient, Secure, and Private Computation.](http://arxiv.org/abs/2204.11168) | 该论文提出了广义拉格朗日编码计算（GLCC）代码，该代码可以提供鲁棒性、安全性和隐私保护的解决方案。通过将数据集分成多个组，使用插值多项式对数据集进行编码，分享编码数据点，可以在主节点上消除跨组的干扰计算结果。GLCC代码是现有拉格朗日编码计算（LCC）代码的一种特例，具有更灵活的折衷方案。 |

# 详细

[^1]: 广义拉格朗日编码计算：用于具有鲁棒性、安全性和隐私保护的计算通信折衷的灵活解决方案

    Generalized Lagrange Coded Computing: A Flexible Computation-Communication Tradeoff for Resilient, Secure, and Private Computation. (arXiv:2204.11168v2 [cs.IT] UPDATED)

    [http://arxiv.org/abs/2204.11168](http://arxiv.org/abs/2204.11168)

    该论文提出了广义拉格朗日编码计算（GLCC）代码，该代码可以提供鲁棒性、安全性和隐私保护的解决方案。通过将数据集分成多个组，使用插值多项式对数据集进行编码，分享编码数据点，可以在主节点上消除跨组的干扰计算结果。GLCC代码是现有拉格朗日编码计算（LCC）代码的一种特例，具有更灵活的折衷方案。

    

    我们考虑在包含多个输入的大规模数据集上评估任意多元多项式的问题，使用一个主节点和多个工作节点的分布式计算系统。提出了广义拉格朗日编码计算（GLCC）代码，同时提供针对不及时返回计算结果的落后者的鲁棒性，针对恶意工人故意修改结果以获取好处的安全性，以及在工人可能共谋的情况下维护数据集的信息理论隐私。GLCC代码首先将数据集分成多个组，然后使用精心设计的插值多项式对数据集进行编码，并将多个编码数据点分享给每个工作器，使得可以在主节点上消除跨组的干扰计算结果。特别地，GLCC代码包括现有最先进的拉格朗日编码计算（LCC）代码作为一种特例，显示出更灵活的折衷方案。

    We consider the problem of evaluating arbitrary multivariate polynomials over a massive dataset containing multiple inputs, on a distributed computing system with a master node and multiple worker nodes. Generalized Lagrange Coded Computing (GLCC) codes are proposed to simultaneously provide resiliency against stragglers who do not return computation results in time, security against adversarial workers who deliberately modify results for their benefit, and information-theoretic privacy of the dataset amidst possible collusion of workers. GLCC codes are constructed by first partitioning the dataset into multiple groups, then encoding the dataset using carefully designed interpolation polynomials, and sharing multiple encoded data points to each worker, such that interference computation results across groups can be eliminated at the master. Particularly, GLCC codes include the state-of-the-art Lagrange Coded Computing (LCC) codes as a special case, and exhibit a more flexible tradeoff 
    

