# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [KTO: Model Alignment as Prospect Theoretic Optimization](https://rss.arxiv.org/abs/2402.01306) | 本文提出了一种名为KTO的方法，将模型对齐视为展望理论优化。与当前方法相比，KTO直接最大化生成效用而不是最大化偏好对数似然。在多个规模上，KTO的性能与基于偏好的方法相当甚至更好。 |
| [^2] | [Provable Multi-Party Reinforcement Learning with Diverse Human Feedback](https://arxiv.org/abs/2403.05006) | 该研究首次提出了多方协作强化学习的理论研究，通过整合多个个体不同偏好的元学习与不同社会福利函数的采用，克服了传统RLHF方法无法捕捉并平衡多个个体偏好的局限性。 |
| [^3] | [High-dimensional Linear Bandits with Knapsacks.](http://arxiv.org/abs/2311.01327) | 本文研究了具有背包约束的高维线性赌臂问题，利用稀疏结构实现改进遗憾。通过开发在线硬阈值算法和原始-对偶框架结合的方法，实现了对特征维度的对数改进的次线性遗憾。 |
| [^4] | [Latent class analysis by regularized spectral clustering.](http://arxiv.org/abs/2310.18727) | 本文提出了两种新的算法用于分类数据的潜在类别模型，并通过新定义的正则化拉普拉斯矩阵计算潜在类别分析，结果表明算法具有理论收敛速度和稳定的一致性。同时提出了一个衡量潜在类别分析强度的度量标准及相关程序，通过模拟实验验证了算法的效率和准确性，并应用于实际分类数据。 |
| [^5] | [Instance-wise Linearization of Neural Network for Model Interpretation.](http://arxiv.org/abs/2310.16295) | 这项研究提出了一种实例化线性化的方法，用于解释神经网络模型。通过给模型内部的每个输入特征分配重要得分，揭示了模型如何使用特征做出决策。这种方法有助于解决当前特征归因方法中的局限性，并提高了模型解释的准确性。 |
| [^6] | [Understanding Uncertainty Sampling.](http://arxiv.org/abs/2307.02719) | 本研究通过系统研究流式和池式主动学习下的不确定性采样算法，提出了一个等效损失的概念，并证明不确定性采样算法实质上是针对该等效损失进行优化。 |

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
    
[^3]: 具有背包约束的高维线性赌臂问题研究

    High-dimensional Linear Bandits with Knapsacks. (arXiv:2311.01327v1 [cs.LG])

    [http://arxiv.org/abs/2311.01327](http://arxiv.org/abs/2311.01327)

    本文研究了具有背包约束的高维线性赌臂问题，利用稀疏结构实现改进遗憾。通过开发在线硬阈值算法和原始-对偶框架结合的方法，实现了对特征维度的对数改进的次线性遗憾。

    

    我们研究了在特征维度较大的高维设置下的具有背包约束的上下文赌臂问题。每个手臂拉动的奖励等于稀疏高维权重向量与当前到达的特征的乘积，加上额外的随机噪声。在本文中，我们研究如何利用这种稀疏结构来实现CBwK问题的改进遗憾。为此，我们首先开发了一种在线的硬阈值算法的变体，以在线方式进行稀疏估计。我们进一步将我们的在线估计器与原始-对偶框架结合起来，在每个背包约束上分配一个对偶变量，并利用在线学习算法来更新对偶变量，从而控制背包容量的消耗。我们证明了这种集成方法使我们能够实现对特征维度的对数改进的次线性遗憾，从而改进了多项式相关性。

    We study the contextual bandits with knapsack (CBwK) problem under the high-dimensional setting where the dimension of the feature is large. The reward of pulling each arm equals the multiplication of a sparse high-dimensional weight vector and the feature of the current arrival, with additional random noise. In this paper, we investigate how to exploit this sparsity structure to achieve improved regret for the CBwK problem. To this end, we first develop an online variant of the hard thresholding algorithm that performs the sparse estimation in an online manner. We further combine our online estimator with a primal-dual framework, where we assign a dual variable to each knapsack constraint and utilize an online learning algorithm to update the dual variable, thereby controlling the consumption of the knapsack capacity. We show that this integrated approach allows us to achieve a sublinear regret that depends logarithmically on the feature dimension, thus improving the polynomial depend
    
[^4]: 通过正则化谱聚类进行潜在类别分析

    Latent class analysis by regularized spectral clustering. (arXiv:2310.18727v1 [cs.LG])

    [http://arxiv.org/abs/2310.18727](http://arxiv.org/abs/2310.18727)

    本文提出了两种新的算法用于分类数据的潜在类别模型，并通过新定义的正则化拉普拉斯矩阵计算潜在类别分析，结果表明算法具有理论收敛速度和稳定的一致性。同时提出了一个衡量潜在类别分析强度的度量标准及相关程序，通过模拟实验验证了算法的效率和准确性，并应用于实际分类数据。

    

    潜在类别模型是在社会、心理和行为科学领域识别具有共同特征的潜在类别的强大工具。在本文中，我们提出了两种新的算法来估计用于分类数据的潜在类别模型。我们的算法利用响应矩阵计算出的新定义的正则化拉普拉斯矩阵进行开发。我们通过考虑稀疏参数给出了算法的理论收敛速度，并表明在适度条件下我们的算法稳定地产生一致的潜在类别分析结果。此外，我们提出了一个衡量潜在类别分析强度的度量标准，并基于该度量标准设计了几个推断在实际分类数据中应该使用多少潜在类别的程序。通过广泛的模拟实验验证了我们算法的效率和准确性，同时将我们的算法进一步应用于实际分类数据。

    The latent class model is a powerful tool for identifying latent classes within populations that share common characteristics for categorical data in social, psychological, and behavioral sciences. In this article, we propose two new algorithms to estimate a latent class model for categorical data. Our algorithms are developed by using a newly defined regularized Laplacian matrix calculated from the response matrix. We provide theoretical convergence rates of our algorithms by considering a sparsity parameter and show that our algorithms stably yield consistent latent class analysis under mild conditions. Additionally, we propose a metric to capture the strength of latent class analysis and several procedures designed based on this metric to infer how many latent classes one should use for real-world categorical data. The efficiency and accuracy of our algorithms are verified by extensive simulated experiments, and we further apply our algorithms to real-world categorical data with pro
    
[^5]: 对于模型解释的神经网络实例化线性化

    Instance-wise Linearization of Neural Network for Model Interpretation. (arXiv:2310.16295v1 [cs.LG])

    [http://arxiv.org/abs/2310.16295](http://arxiv.org/abs/2310.16295)

    这项研究提出了一种实例化线性化的方法，用于解释神经网络模型。通过给模型内部的每个输入特征分配重要得分，揭示了模型如何使用特征做出决策。这种方法有助于解决当前特征归因方法中的局限性，并提高了模型解释的准确性。

    

    神经网络在许多科学领域取得了显著的成功。然而，神经网络模型的可解释性仍然是将这种技术应用于我们日常生活的主要瓶颈。挑战在于神经网络的非线性行为，这提出了一个关键性问题，即模型如何使用输入特征进行决策。解决这一挑战的经典方法是特征归因，它为每个输入特征分配一个重要得分，并揭示其对当前预测的重要性。然而，当前的特征归因方法经常指示每个输入特征的重要性，而没有详细说明它们在模型内部实际上是如何处理的。这些归因方法常常引发一个关注点，即它们是否正确地强调了模型预测的特征。对于神经网络模型，非线性行为通常是由模型的非线性激活单元引起的。然而，预测的计算行为往往是复杂的，这使得解释和理解模型的决策变得困难。

    Neural network have achieved remarkable successes in many scientific fields. However, the interpretability of the neural network model is still a major bottlenecks to deploy such technique into our daily life. The challenge can dive into the non-linear behavior of the neural network, which rises a critical question that how a model use input feature to make a decision. The classical approach to address this challenge is feature attribution, which assigns an important score to each input feature and reveal its importance of current prediction. However, current feature attribution approaches often indicate the importance of each input feature without detail of how they are actually processed by a model internally. These attribution approaches often raise a concern that whether they highlight correct features for a model prediction.  For a neural network model, the non-linear behavior is often caused by non-linear activation units of a model. However, the computation behavior of a predict
    
[^6]: 理解不确定性采样

    Understanding Uncertainty Sampling. (arXiv:2307.02719v1 [cs.LG])

    [http://arxiv.org/abs/2307.02719](http://arxiv.org/abs/2307.02719)

    本研究通过系统研究流式和池式主动学习下的不确定性采样算法，提出了一个等效损失的概念，并证明不确定性采样算法实质上是针对该等效损失进行优化。

    

    不确定性采样是一种常见的主动学习算法，它顺序地查询当前预测模型对数据样本的不确定性。然而，不确定性采样的使用往往是启发式的：（i）关于在特定任务和特定损失函数下对“不确定性”的准确定义没有共识；（ii）没有理论保证能够给出一个标准协议来实施该算法，例如，在随机梯度下降等优化算法框架下如何处理顺序到达的注释数据。在本研究中，我们系统地研究了流式和池式主动学习下的不确定性采样算法。我们提出了一个等效损失的概念，该概念取决于使用的不确定性度量和原始损失函数，并确立了不确定性采样算法本质上是针对这种等效损失进行优化。这一观点验证了算法的适当性。

    Uncertainty sampling is a prevalent active learning algorithm that queries sequentially the annotations of data samples which the current prediction model is uncertain about. However, the usage of uncertainty sampling has been largely heuristic: (i) There is no consensus on the proper definition of "uncertainty" for a specific task under a specific loss; (ii) There is no theoretical guarantee that prescribes a standard protocol to implement the algorithm, for example, how to handle the sequentially arrived annotated data under the framework of optimization algorithms such as stochastic gradient descent. In this work, we systematically examine uncertainty sampling algorithms under both stream-based and pool-based active learning. We propose a notion of equivalent loss which depends on the used uncertainty measure and the original loss function and establish that an uncertainty sampling algorithm essentially optimizes against such an equivalent loss. The perspective verifies the properne
    

