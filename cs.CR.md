# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Differentially Private Model-Based Offline Reinforcement Learning](https://arxiv.org/abs/2402.05525) | 本研究提出了一种差分隐私的基于模型的离线强化学习方法，通过学习离线数据中的隐私模型以及基于模型的策略优化，实现了从离线数据中训练具有隐私保护的强化学习代理。同时，研究还总结了在这种设置下隐私的代价。 |
| [^2] | [Kick Bad Guys Out! Zero-Knowledge-Proof-Based Anomaly Detection in Federated Learning.](http://arxiv.org/abs/2310.04055) | 本文提出了一种基于零知识证明的联邦学习异常检测方法，实现了在实际系统中检测和消除恶意客户端模型的能力。 |
| [^3] | [Practical, Private Assurance of the Value of Collaboration.](http://arxiv.org/abs/2310.02563) | 该论文研究了两方在数据集上合作前如何保证合作的价值。通过构建基于全同态加密方案和标签差分隐私的交互式协议，该研究提供了一个实用的、私密的解决方案。最终的结果是确保合作前双方的模型和数据集不会被透露。 |

# 详细

[^1]: 差分隐私的基于模型的离线强化学习

    Differentially Private Model-Based Offline Reinforcement Learning

    [https://arxiv.org/abs/2402.05525](https://arxiv.org/abs/2402.05525)

    本研究提出了一种差分隐私的基于模型的离线强化学习方法，通过学习离线数据中的隐私模型以及基于模型的策略优化，实现了从离线数据中训练具有隐私保护的强化学习代理。同时，研究还总结了在这种设置下隐私的代价。

    

    我们解决了具有隐私保证的离线强化学习问题，目标是训练一个相对于数据集中每个轨迹具有差分隐私的策略。为了实现这一目标，我们引入了DP-MORL，一种带有差分隐私保证的MBRL算法。首先，使用DP-FedAvg从离线数据中学习环境的隐私模型，DP-FedAvg是一种为神经网络提供轨迹级差分隐私保证的训练方法。然后，我们使用基于模型的策略优化从（受罚的）隐私模型中推导出策略，无需进一步与系统交互或访问输入数据。我们经验证明，DP-MORL能够从离线数据中训练出具有隐私保护的RL代理，并进一步概述了在这种情况下隐私的代价。

    We address offline reinforcement learning with privacy guarantees, where the goal is to train a policy that is differentially private with respect to individual trajectories in the dataset. To achieve this, we introduce DP-MORL, an MBRL algorithm coming with differential privacy guarantees. A private model of the environment is first learned from offline data using DP-FedAvg, a training method for neural networks that provides differential privacy guarantees at the trajectory level. Then, we use model-based policy optimization to derive a policy from the (penalized) private model, without any further interaction with the system or access to the input data. We empirically show that DP-MORL enables the training of private RL agents from offline data and we furthermore outline the price of privacy in this setting.
    
[^2]: 把坏人踢出去！基于零知识证明的联邦学习异常检测

    Kick Bad Guys Out! Zero-Knowledge-Proof-Based Anomaly Detection in Federated Learning. (arXiv:2310.04055v1 [cs.CR])

    [http://arxiv.org/abs/2310.04055](http://arxiv.org/abs/2310.04055)

    本文提出了一种基于零知识证明的联邦学习异常检测方法，实现了在实际系统中检测和消除恶意客户端模型的能力。

    

    联邦学习系统容易受到恶意客户端的攻击，他们通过提交篡改的本地模型来达到对抗目标，比如阻止全局模型的收敛或者导致全局模型对某些数据进行错误分类。许多现有的防御机制在实际联邦学习系统中不可行，因为它们需要先知道恶意客户端的数量，或者依赖重新加权或修改提交的方式。这是因为攻击者通常不会在攻击之前宣布他们的意图，而重新加权可能会改变聚合结果，即使没有攻击。为了解决这些在实际联邦学习系统中的挑战，本文引入了一种最尖端的异常检测方法，具有以下特点：i）仅在发生攻击时检测攻击的发生并进行防御操作；ii）一旦发生攻击，进一步检测恶意客户端模型并将其消除，而不会对正常模型造成伤害；iii）确保

    Federated learning (FL) systems are vulnerable to malicious clients that submit poisoned local models to achieve their adversarial goals, such as preventing the convergence of the global model or inducing the global model to misclassify some data. Many existing defense mechanisms are impractical in real-world FL systems, as they require prior knowledge of the number of malicious clients or rely on re-weighting or modifying submissions. This is because adversaries typically do not announce their intentions before attacking, and re-weighting might change aggregation results even in the absence of attacks. To address these challenges in real FL systems, this paper introduces a cutting-edge anomaly detection approach with the following features: i) Detecting the occurrence of attacks and performing defense operations only when attacks happen; ii) Upon the occurrence of an attack, further detecting the malicious client models and eliminating them without harming the benign ones; iii) Ensuri
    
[^3]: 实用的、私密的合作价值保证

    Practical, Private Assurance of the Value of Collaboration. (arXiv:2310.02563v1 [cs.CR])

    [http://arxiv.org/abs/2310.02563](http://arxiv.org/abs/2310.02563)

    该论文研究了两方在数据集上合作前如何保证合作的价值。通过构建基于全同态加密方案和标签差分隐私的交互式协议，该研究提供了一个实用的、私密的解决方案。最终的结果是确保合作前双方的模型和数据集不会被透露。

    

    两个方向希望在数据集上进行合作。然而，在彼此透露数据集之前，双方希望能够得到合作将是富有成果的保证。我们从机器学习的角度来看待这个问题，其中一方被承诺通过合并来自另一方的数据来改进其预测模型。只有当更新的模型显示出准确性的提升时，双方才希望进一步合作。在确定这一点之前，双方不希望透露他们的模型和数据集。在这项工作中，我们基于Torus上的全同态加密方案（TFHE）和标签差分隐私构建了一个交互式协议，其中底层的机器学习模型是一个神经网络。标签差分隐私用于确保计算不完全在加密领域进行，这对神经网络训练来说是一个重要瓶颈。

    Two parties wish to collaborate on their datasets. However, before they reveal their datasets to each other, the parties want to have the guarantee that the collaboration would be fruitful. We look at this problem from the point of view of machine learning, where one party is promised an improvement on its prediction model by incorporating data from the other party. The parties would only wish to collaborate further if the updated model shows an improvement in accuracy. Before this is ascertained, the two parties would not want to disclose their models and datasets. In this work, we construct an interactive protocol for this problem based on the fully homomorphic encryption scheme over the Torus (TFHE) and label differential privacy, where the underlying machine learning model is a neural network. Label differential privacy is used to ensure that computations are not done entirely in the encrypted domain, which is a significant bottleneck for neural network training according to the cu
    

