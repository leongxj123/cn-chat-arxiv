# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Obtaining Explainable Classification Models using Distributionally Robust Optimization.](http://arxiv.org/abs/2311.01994) | 本论文介绍了一种利用分布鲁棒优化获取可解释的分类模型的方法，通过构建稀疏的规则集合来同时解决规则集的稀疏性和预测准确性之间的权衡，从而保证泛化性能并降低计算成本。 |
| [^2] | [Machine Learning Infused Distributed Optimization for Coordinating Virtual Power Plant Assets.](http://arxiv.org/abs/2310.17882) | 本文提出了一种名为LOOP-MAC的方法，利用机器学习辅助的分布式优化来协调虚拟电厂（VPP）的资产。该方法采用多智能体协调视角，利用神经网络逼近器加速解决方案搜索。该方法通过引入测量图来保证最佳解之间的关系。 |

# 详细

[^1]: 利用分布鲁棒优化获取可解释的分类模型

    Obtaining Explainable Classification Models using Distributionally Robust Optimization. (arXiv:2311.01994v1 [stat.ML])

    [http://arxiv.org/abs/2311.01994](http://arxiv.org/abs/2311.01994)

    本论文介绍了一种利用分布鲁棒优化获取可解释的分类模型的方法，通过构建稀疏的规则集合来同时解决规则集的稀疏性和预测准确性之间的权衡，从而保证泛化性能并降低计算成本。

    

    对于人类用户来说，模型的可解释性对于理解提议分类器如何根据特征值给数据分配标签至关重要。我们研究使用特征值规则集构建的广义线性模型，该模型可以捕捉非线性依赖和交互作用。规则集的稀疏性和预测准确性之间存在固有的权衡。使用现有方法来找到合适的稀疏度选择（例如通过交叉验证）计算成本很高。我们提出了一种新的公式来学习同时解决这些竞争因素的规则集合。通过利用分布鲁棒优化来确保良好的泛化性能，同时保持低计算成本。该公式利用列生成有效地搜索规则集合的空间并构建稀疏的规则集合，与随机森林或Boosting及其变体等技术相比。我们提出了理论结果来推动这一公式的发展。

    Model explainability is crucial for human users to be able to interpret how a proposed classifier assigns labels to data based on its feature values. We study generalized linear models constructed using sets of feature value rules, which can capture nonlinear dependencies and interactions. An inherent trade-off exists between rule set sparsity and its prediction accuracy. It is computationally expensive to find the right choice of sparsity -- e.g., via cross-validation -- with existing methods. We propose a new formulation to learn an ensemble of rule sets that simultaneously addresses these competing factors. Good generalization is ensured while keeping computational costs low by utilizing distributionally robust optimization. The formulation utilizes column generation to efficiently search the space of rule sets and constructs a sparse ensemble of rule sets, in contrast with techniques like random forests or boosting and their variants. We present theoretical results that motivate an
    
[^2]: 为协调虚拟电厂资产的分布式优化引入机器学习

    Machine Learning Infused Distributed Optimization for Coordinating Virtual Power Plant Assets. (arXiv:2310.17882v1 [cs.LG])

    [http://arxiv.org/abs/2310.17882](http://arxiv.org/abs/2310.17882)

    本文提出了一种名为LOOP-MAC的方法，利用机器学习辅助的分布式优化来协调虚拟电厂（VPP）的资产。该方法采用多智能体协调视角，利用神经网络逼近器加速解决方案搜索。该方法通过引入测量图来保证最佳解之间的关系。

    

    随着对分布式能源资源（DER）部署日益增加的兴趣，虚拟电厂（VPP）已成为汇集各种DER并促进其参与批发能源市场的关键工具。这些VPP部署得到了联邦能源监管委员会第2222号命令的推动，该命令使DER和VPP在市场领域具有竞争力。然而，DER的多样性和分散性质给VPP资产的可扩展协调带来了显著挑战。为了解决效率和速度瓶颈，本文提出了一种新颖的机器学习辅助的分布式优化方法来协调VPP资产。我们的方法名为LOOP-MAC（为多智能体协调优化过程学习优化），采用多智能体协调视角，每个VPP代理管理多个DER并利用神经网络逼近器加速解决方案搜索。LOOP-MAC方法使用一个测量图确保了最佳解之间的关系。

    Amid the increasing interest in the deployment of Distributed Energy Resources (DERs), the Virtual Power Plant (VPP) has emerged as a pivotal tool for aggregating diverse DERs and facilitating their participation in wholesale energy markets. These VPP deployments have been fueled by the Federal Energy Regulatory Commission's Order 2222, which makes DERs and VPPs competitive across market segments. However, the diversity and decentralized nature of DERs present significant challenges to the scalable coordination of VPP assets. To address efficiency and speed bottlenecks, this paper presents a novel machine learning-assisted distributed optimization to coordinate VPP assets. Our method, named LOOP-MAC(Learning to Optimize the Optimization Process for Multi-agent Coordination), adopts a multi-agent coordination perspective where each VPP agent manages multiple DERs and utilizes neural network approximators to expedite the solution search. The LOOP-MAC method employs a gauge map to guarant
    

