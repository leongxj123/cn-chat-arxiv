# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Noise Contrastive Estimation-based Matching Framework for Low-resource Security Attack Pattern Recognition.](http://arxiv.org/abs/2401.10337) | 该论文提出了一种基于噪声对比估计的低资源安全攻击模式识别匹配框架，通过直接语义相似度决定文本与攻击模式之间的关联，以降低大量类别、标签分布不均和标签空间复杂性带来的学习难度。 |
| [^2] | [Theoretically Principled Federated Learning for Balancing Privacy and Utility.](http://arxiv.org/abs/2305.15148) | 本文提出基于扭曲模型参数的保护机制的通用学习框架，用于实现联合学习中隐私保护和数据效用的平衡。算法可以在每个通信轮中实现个性化的效用-隐私折衷，我们在理论上证明了算法的次线性性质，该算法可以提高隐私保护的联合学习效力。 |

# 详细

[^1]: 基于噪声对比估计的低资源安全攻击模式识别匹配框架

    Noise Contrastive Estimation-based Matching Framework for Low-resource Security Attack Pattern Recognition. (arXiv:2401.10337v1 [cs.LG])

    [http://arxiv.org/abs/2401.10337](http://arxiv.org/abs/2401.10337)

    该论文提出了一种基于噪声对比估计的低资源安全攻击模式识别匹配框架，通过直接语义相似度决定文本与攻击模式之间的关联，以降低大量类别、标签分布不均和标签空间复杂性带来的学习难度。

    

    战术、技术和程序（TTPs）是网络安全领域中复杂的攻击模式，在文本知识库中有详细的描述。在网络安全写作中识别TTPs，通常称为TTP映射，是一个重要而具有挑战性的任务。传统的学习方法通常以经典的多类或多标签分类设置为目标。由于存在大量的类别（即TTPs），标签分布的不均衡和标签空间的复杂层次结构，这种设置限制了模型的学习能力。我们采用了一种不同的学习范式来解决这个问题，其中将文本与TTP标签之间的直接语义相似度决定为文本分配给TTP标签，从而减少了仅仅在大型标签空间上竞争的复杂性。为此，我们提出了一种具有有效的基于采样的学习比较机制的神经匹配架构，促进学习过程。

    Tactics, Techniques and Procedures (TTPs) represent sophisticated attack patterns in the cybersecurity domain, described encyclopedically in textual knowledge bases. Identifying TTPs in cybersecurity writing, often called TTP mapping, is an important and challenging task. Conventional learning approaches often target the problem in the classical multi-class or multilabel classification setting. This setting hinders the learning ability of the model due to a large number of classes (i.e., TTPs), the inevitable skewness of the label distribution and the complex hierarchical structure of the label space. We formulate the problem in a different learning paradigm, where the assignment of a text to a TTP label is decided by the direct semantic similarity between the two, thus reducing the complexity of competing solely over the large labeling space. To that end, we propose a neural matching architecture with an effective sampling-based learn-to-compare mechanism, facilitating the learning pr
    
[^2]: 理论指导的联合学习实现了隐私保护和数据效用的平衡

    Theoretically Principled Federated Learning for Balancing Privacy and Utility. (arXiv:2305.15148v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.15148](http://arxiv.org/abs/2305.15148)

    本文提出基于扭曲模型参数的保护机制的通用学习框架，用于实现联合学习中隐私保护和数据效用的平衡。算法可以在每个通信轮中实现个性化的效用-隐私折衷，我们在理论上证明了算法的次线性性质，该算法可以提高隐私保护的联合学习效力。

    

    我们提出了一种保护机制的通用学习框架，通过扭曲模型参数来保护隐私，实现隐私和效用之间的平衡。该算法适用于任意将扭曲映射到实值的隐私测量。在联合学习中，它可以为每个模型参数，每个客户端，在每个通信轮中实现个性化的效用-隐私折衷。这种自适应和细粒度的保护可以提高保护隐私的联合学习的效力。从理论上讲，我们证明了算法保护超参数的效用损失与最优保护超参数的效用损失之间的差距是总迭代次数的次线性。我们的算法次线性的特点表明，当迭代次数趋近于无穷大时，算法性能和最优性能之间的平均差距趋近于零。此外，我们提供了收敛性证明。

    We propose a general learning framework for the protection mechanisms that protects privacy via distorting model parameters, which facilitates the trade-off between privacy and utility. The algorithm is applicable to arbitrary privacy measurements that maps from the distortion to a real value. It can achieve personalized utility-privacy trade-off for each model parameter, on each client, at each communication round in federated learning. Such adaptive and fine-grained protection can improve the effectiveness of privacy-preserved federated learning.  Theoretically, we show that gap between the utility loss of the protection hyperparameter output by our algorithm and that of the optimal protection hyperparameter is sub-linear in the total number of iterations. The sublinearity of our algorithm indicates that the average gap between the performance of our algorithm and that of the optimal performance goes to zero when the number of iterations goes to infinity. Further, we provide the conv
    

