# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [KTO: Model Alignment as Prospect Theoretic Optimization](https://rss.arxiv.org/abs/2402.01306) | 本文提出了一种名为KTO的方法，将模型对齐视为展望理论优化。与当前方法相比，KTO直接最大化生成效用而不是最大化偏好对数似然。在多个规模上，KTO的性能与基于偏好的方法相当甚至更好。 |
| [^2] | [Provable Multi-Party Reinforcement Learning with Diverse Human Feedback](https://arxiv.org/abs/2403.05006) | 该研究首次提出了多方协作强化学习的理论研究，通过整合多个个体不同偏好的元学习与不同社会福利函数的采用，克服了传统RLHF方法无法捕捉并平衡多个个体偏好的局限性。 |
| [^3] | [Instance-wise Linearization of Neural Network for Model Interpretation.](http://arxiv.org/abs/2310.16295) | 这项研究提出了一种实例化线性化的方法，用于解释神经网络模型。通过给模型内部的每个输入特征分配重要得分，揭示了模型如何使用特征做出决策。这种方法有助于解决当前特征归因方法中的局限性，并提高了模型解释的准确性。 |
| [^4] | [Topological Guided Actor-Critic Modular Learning of Continuous Systems with Temporal Objectives.](http://arxiv.org/abs/2304.10041) | 本文介绍了一种基于拓扑指导的 Actor-Critic 模块化学习方法，用于解决连续系统中的最优策略综合问题，并提出了一种广义的最优备份顺序来加速学习过程。 |

# 详细

[^1]: KTO: 模型对齐视为展望理论优化

    KTO: Model Alignment as Prospect Theoretic Optimization

    [https://rss.arxiv.org/abs/2402.01306](https://rss.arxiv.org/abs/2402.01306)

    本文提出了一种名为KTO的方法，将模型对齐视为展望理论优化。与当前方法相比，KTO直接最大化生成效用而不是最大化偏好对数似然。在多个规模上，KTO的性能与基于偏好的方法相当甚至更好。

    

    凯恩曼与特沃斯基的展望理论告诉我们，人类以有偏见但明确的方式看待随机变量；例如，人们通常都是厌恶损失的。我们证明了将LLMs与人工反馈进行对齐的目标隐含地融合了许多这些偏见 - 这些目标 (例如 DPO) 的成功部分可归因于它们是"人类感知损失函数"(HALOs)。然而，这些方法所归因给人类的效用函数仍与展望理论文献中的不同。利用凯恩曼-特沃斯基人类效用的模型，我们提出了一种直接最大化生成效用而不是最大化偏好对数似然的HALO。我们将这种方法称为凯恩曼-特沃斯基优化(KTO)，并且它在从1B到30B的规模上与基于偏好的方法的性能相匹配或超过。关键是，KTO不需要偏好 - 只需要一个是否的二进制信号。

    Kahneman & Tversky's $\textit{prospect theory}$ tells us that humans perceive random variables in a biased but well-defined manner; for example, humans are famously loss-averse. We show that objectives for aligning LLMs with human feedback implicitly incorporate many of these biases -- the success of these objectives (e.g., DPO) over cross-entropy minimization can partly be ascribed to them being $\textit{human-aware loss functions}$ (HALOs). However, the utility functions these methods attribute to humans still differ from those in the prospect theory literature. Using a Kahneman-Tversky model of human utility, we propose a HALO that directly maximizes the utility of generations instead of maximizing the log-likelihood of preferences, as current methods do. We call this approach Kahneman-Tversky Optimization (KTO), and it matches or exceeds the performance of preference-based methods at scales from 1B to 30B. Crucially, KTO does not need preferences -- only a binary signal of whether 
    
[^2]: 具有多元人类反馈的可证明多方协作强化学习

    Provable Multi-Party Reinforcement Learning with Diverse Human Feedback

    [https://arxiv.org/abs/2403.05006](https://arxiv.org/abs/2403.05006)

    该研究首次提出了多方协作强化学习的理论研究，通过整合多个个体不同偏好的元学习与不同社会福利函数的采用，克服了传统RLHF方法无法捕捉并平衡多个个体偏好的局限性。

    

    用人类反馈进行强化学习（RLHF）是一种新兴范式，旨在将模型与人类偏好进行匹配。我们的工作探索了明确建模多个个体不同偏好的多方RLHF的理论研究。我们展示了传统RLHF方法如何失败，因为学习单一奖励函数无法捕捉和平衡多个个体的偏好。为了克服这些局限性，我们结合元学习来学习多个偏好，并采用不同的社会福利函数来整合多方的偏好。我们关注离线学习设置，并为优化不同社会福利函数（如Nash、Utilitarian和Leximin福利）建立样本复杂度界限，同时提供效率和公平性保证。

    arXiv:2403.05006v1 Announce Type: cross  Abstract: Reinforcement learning with human feedback (RLHF) is an emerging paradigm to align models with human preferences. Typically, RLHF aggregates preferences from multiple individuals who have diverse viewpoints that may conflict with each other. Our work \textit{initiates} the theoretical study of multi-party RLHF that explicitly models the diverse preferences of multiple individuals. We show how traditional RLHF approaches can fail since learning a single reward function cannot capture and balance the preferences of multiple individuals. To overcome such limitations, we incorporate meta-learning to learn multiple preferences and adopt different social welfare functions to aggregate the preferences across multiple parties. We focus on the offline learning setting and establish sample complexity bounds, along with efficiency and fairness guarantees, for optimizing diverse social welfare functions such as Nash, Utilitarian, and Leximin welfa
    
[^3]: 对于模型解释的神经网络实例化线性化

    Instance-wise Linearization of Neural Network for Model Interpretation. (arXiv:2310.16295v1 [cs.LG])

    [http://arxiv.org/abs/2310.16295](http://arxiv.org/abs/2310.16295)

    这项研究提出了一种实例化线性化的方法，用于解释神经网络模型。通过给模型内部的每个输入特征分配重要得分，揭示了模型如何使用特征做出决策。这种方法有助于解决当前特征归因方法中的局限性，并提高了模型解释的准确性。

    

    神经网络在许多科学领域取得了显著的成功。然而，神经网络模型的可解释性仍然是将这种技术应用于我们日常生活的主要瓶颈。挑战在于神经网络的非线性行为，这提出了一个关键性问题，即模型如何使用输入特征进行决策。解决这一挑战的经典方法是特征归因，它为每个输入特征分配一个重要得分，并揭示其对当前预测的重要性。然而，当前的特征归因方法经常指示每个输入特征的重要性，而没有详细说明它们在模型内部实际上是如何处理的。这些归因方法常常引发一个关注点，即它们是否正确地强调了模型预测的特征。对于神经网络模型，非线性行为通常是由模型的非线性激活单元引起的。然而，预测的计算行为往往是复杂的，这使得解释和理解模型的决策变得困难。

    Neural network have achieved remarkable successes in many scientific fields. However, the interpretability of the neural network model is still a major bottlenecks to deploy such technique into our daily life. The challenge can dive into the non-linear behavior of the neural network, which rises a critical question that how a model use input feature to make a decision. The classical approach to address this challenge is feature attribution, which assigns an important score to each input feature and reveal its importance of current prediction. However, current feature attribution approaches often indicate the importance of each input feature without detail of how they are actually processed by a model internally. These attribution approaches often raise a concern that whether they highlight correct features for a model prediction.  For a neural network model, the non-linear behavior is often caused by non-linear activation units of a model. However, the computation behavior of a predict
    
[^4]: 面向具有时间目标的连续系统的拓扑指导的 Actor-Critic 模块化学习

    Topological Guided Actor-Critic Modular Learning of Continuous Systems with Temporal Objectives. (arXiv:2304.10041v1 [cs.AI])

    [http://arxiv.org/abs/2304.10041](http://arxiv.org/abs/2304.10041)

    本文介绍了一种基于拓扑指导的 Actor-Critic 模块化学习方法，用于解决连续系统中的最优策略综合问题，并提出了一种广义的最优备份顺序来加速学习过程。

    

    本文研究了在线性时间逻辑中给定高级规范的连续状态随机动态系统的正式策略综合。为了学习最大化满足概率的最优策略，我们对动态系统和翻译出的自动机进行了乘积构造，从而构建了一个乘积系统，然后在其中解决了最优计划问题。由于乘积系统具有混合乘积状态空间，导致奖励稀疏，因此我们引入了一种广义的最优备份顺序，与拓扑顺序相反，以指导值备份并加速学习过程。我们提供了使用广义最优备份顺序进行最优计划问题的最优性证明。此外，本文提出了一种基于拓扑顺序的演员-评论家强化学习算法。该算法利用先进的数学技术，并具有超参数自调整的特性。我们提供了该算法的最优性和收敛性证明。

    This work investigates the formal policy synthesis of continuous-state stochastic dynamic systems given high-level specifications in linear temporal logic. To learn an optimal policy that maximizes the satisfaction probability, we take a product between a dynamic system and the translated automaton to construct a product system on which we solve an optimal planning problem. Since this product system has a hybrid product state space that results in reward sparsity, we introduce a generalized optimal backup order, in reverse to the topological order, to guide the value backups and accelerate the learning process. We provide the optimality proof for using the generalized optimal backup order in this optimal planning problem. Further, this paper presents an actor-critic reinforcement learning algorithm when topological order applies. This algorithm leverages advanced mathematical techniques and enjoys the property of hyperparameter self-tuning. We provide proof of the optimality and conver
    

