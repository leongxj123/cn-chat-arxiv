# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Exploiting Symmetry in Dynamics for Model-Based Reinforcement Learning with Asymmetric Rewards](https://arxiv.org/abs/2403.19024) | 本文扩展了强化学习和控制理论中对称技术的应用范围，通过利用动态对称性学习动力学模型，而不要求奖励具有相同的对称性。 |
| [^2] | [Conjectural Online Learning with First-order Beliefs in Asymmetric Information Stochastic Games](https://arxiv.org/abs/2402.18781) | 提出了一种具有假设在线学习（COL）的学习方案，针对通用AISG，结构化为一个先验预测者-演员-评论家（FAC）架构，利用一级信念和对手策略的主观预测，通过在线展开更新策略，并通过贝叶斯学习校准假设。 |
| [^3] | [PowerGraph: A power grid benchmark dataset for graph neural networks](https://arxiv.org/abs/2402.02827) | PowerGraph是一个用于图神经网络的电网基准数据集，旨在通过机器学习模型实现电力网格断电的在线检测。 |
| [^4] | [A Generalizable Physics-informed Learning Framework for Risk Probability Estimation.](http://arxiv.org/abs/2305.06432) | 本文提出了一种基于物理学的学习框架，通过将MC方法与基于物理学的神经网络相结合，有效评估长期风险概率及其梯度。数值结果表明，该方法具有更好的样本效率，能够适应系统变化。 |

# 详细

[^1]: 利用动态对称性进行基于模型的非对称奖励强化学习

    Exploiting Symmetry in Dynamics for Model-Based Reinforcement Learning with Asymmetric Rewards

    [https://arxiv.org/abs/2403.19024](https://arxiv.org/abs/2403.19024)

    本文扩展了强化学习和控制理论中对称技术的应用范围，通过利用动态对称性学习动力学模型，而不要求奖励具有相同的对称性。

    

    强化学习中最近的工作利用模型中的对称性来提高策略训练的采样效率。一个常用的简化假设是动力学和奖励都表现出相同的对称性。然而，在许多真实环境中，动力学模型表现出与奖励模型独立的对称性：奖励可能不满足与动力学相同的对称性。本文探讨了只假定动力学表现出对称性的情况，扩展了强化学习和控制理论学习中可应用对称技术的问题范围。我们利用卡塔恩移动框架方法引入一种学习动力学的技术，通过构造，这种动力学表现出指定的对称性。我们通过数值实验展示了所提出的方法学到了更准确的动态模型。

    arXiv:2403.19024v1 Announce Type: cross  Abstract: Recent work in reinforcement learning has leveraged symmetries in the model to improve sample efficiency in training a policy. A commonly used simplifying assumption is that the dynamics and reward both exhibit the same symmetry. However, in many real-world environments, the dynamical model exhibits symmetry independent of the reward model: the reward may not satisfy the same symmetries as the dynamics. In this paper, we investigate scenarios where only the dynamics are assumed to exhibit symmetry, extending the scope of problems in reinforcement learning and learning in control theory where symmetry techniques can be applied. We use Cartan's moving frame method to introduce a technique for learning dynamics which, by construction, exhibit specified symmetries. We demonstrate through numerical experiments that the proposed method learns a more accurate dynamical model.
    
[^2]: 具有一级信念的假设在线学习在不对称信息随机博弈中的应用

    Conjectural Online Learning with First-order Beliefs in Asymmetric Information Stochastic Games

    [https://arxiv.org/abs/2402.18781](https://arxiv.org/abs/2402.18781)

    提出了一种具有假设在线学习（COL）的学习方案，针对通用AISG，结构化为一个先验预测者-演员-评论家（FAC）架构，利用一级信念和对手策略的主观预测，通过在线展开更新策略，并通过贝叶斯学习校准假设。

    

    随机博弈出现在许多复杂的社会技术系统中，如网络物理系统和IT基础设施，信息不对称为决策实体（玩家）的决策带来挑战。现有的不对称信息随机博弈（AISG）的计算方法主要是离线的，针对特殊类别的AISG，以避免信念层次，并且缺乏适应均衡偏差的在线能力。为了解决这一限制，我们提出了一种具有假设在线学习（COL）的学习方案，专门针对通用AISG。COL结构化为一个先验预测者-演员-评论家（FAC）架构，利用对隐藏状态的一级信念和对对手策略的主观预测。针对假设的对手，COL通过在线展开更新策略，并通过贝叶斯学习校准假设。我们证明了COL中的假设与t一致。

    arXiv:2402.18781v1 Announce Type: cross  Abstract: Stochastic games arise in many complex socio-technical systems, such as cyber-physical systems and IT infrastructures, where information asymmetry presents challenges for decision-making entities (players). Existing computational methods for asymmetric information stochastic games (AISG) are primarily offline, targeting special classes of AISGs to avoid belief hierarchies, and lack online adaptability to deviations from equilibrium. To address this limitation, we propose a conjectural online learning (COL), a learning scheme for generic AISGs. COL, structured as a forecaster-actor-critic (FAC) architecture, utilizes first-order beliefs over the hidden states and subjective forecasts of the opponent's strategies. Against the conjectured opponent, COL updates strategies in an actor-critic approach using online rollout and calibrates conjectures through Bayesian learning. We prove that conjecture in COL is asymptotically consistent with t
    
[^3]: PowerGraph: 用于图神经网络的电网基准数据集

    PowerGraph: A power grid benchmark dataset for graph neural networks

    [https://arxiv.org/abs/2402.02827](https://arxiv.org/abs/2402.02827)

    PowerGraph是一个用于图神经网络的电网基准数据集，旨在通过机器学习模型实现电力网格断电的在线检测。

    

    公共图神经网络（GNN）基准数据集有助于使用GNN，并增强GNN在各个领域中的适用性。目前，社区中缺乏用于GNN应用的电力网格公共数据集。事实上，与其他机器学习技术相比，GNN可以潜在地捕捉到复杂的电力网格现象。电力网格是复杂的工程网络，天然适合于图表示。因此，GNN有潜力捕捉到电力网格的行为，而不用其他机器学习技术。为了实现这个目标，我们开发了一个用于级联故障事件的图数据集，这是导致电力网格断电的主要原因。历史断电数据集稀缺且不完整。通常通过计算昂贵的离线级联故障模拟来评估脆弱性和识别关键组件。相反，我们建议使用机器学习模型进行在线检测。

    Public Graph Neural Networks (GNN) benchmark datasets facilitate the use of GNN and enhance GNN applicability to diverse disciplines. The community currently lacks public datasets of electrical power grids for GNN applications. Indeed, GNNs can potentially capture complex power grid phenomena over alternative machine learning techniques. Power grids are complex engineered networks that are naturally amenable to graph representations. Therefore, GNN have the potential for capturing the behavior of power grids over alternative machine learning techniques. To this aim, we develop a graph dataset for cascading failure events, which are the major cause of blackouts in electric power grids. Historical blackout datasets are scarce and incomplete. The assessment of vulnerability and the identification of critical components are usually conducted via computationally expensive offline simulations of cascading failures. Instead, we propose using machine learning models for the online detection of
    
[^4]: 一种适用于风险概率估计的可推广、物理学基础学习框架

    A Generalizable Physics-informed Learning Framework for Risk Probability Estimation. (arXiv:2305.06432v1 [eess.SY])

    [http://arxiv.org/abs/2305.06432](http://arxiv.org/abs/2305.06432)

    本文提出了一种基于物理学的学习框架，通过将MC方法与基于物理学的神经网络相结合，有效评估长期风险概率及其梯度。数值结果表明，该方法具有更好的样本效率，能够适应系统变化。

    

    准确评估长期风险概率及其梯度对于许多随机安全控制方法至关重要。然而，在实时和未知或变化的环境中计算这些风险概率是具有挑战性的。在本文中，我们开发了一种有效的方法来评估长期风险概率及其梯度。所提出的方法利用了长期风险概率满足某些偏微分方程(PDEs)的事实，该方程表征了概率之间的邻近关系，以将MC方法和基于物理学的神经网络相结合。我们提供了在特定训练配置下给出估计误差的理论保证。数值结果表明，所提出的方法具有更好的样本效率，能够很好地推广到未知区域，并能够适应系统变化，相比MC方法和现有的数据驱动方法，它表现出更好的性能。

    Accurate estimates of long-term risk probabilities and their gradients are critical for many stochastic safe control methods. However, computing such risk probabilities in real-time and in unseen or changing environments is challenging. Monte Carlo (MC) methods cannot accurately evaluate the probabilities and their gradients as an infinitesimal devisor can amplify the sampling noise. In this paper, we develop an efficient method to evaluate the probabilities of long-term risk and their gradients. The proposed method exploits the fact that long-term risk probability satisfies certain partial differential equations (PDEs), which characterize the neighboring relations between the probabilities, to integrate MC methods and physics-informed neural networks. We provide theoretical guarantees of the estimation error given certain choices of training configurations. Numerical results show the proposed method has better sample efficiency, generalizes well to unseen regions, and can adapt to sys
    

