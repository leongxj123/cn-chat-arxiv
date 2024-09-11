# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-intention Inverse Q-learning for Interpretable Behavior Representation](https://rss.arxiv.org/abs/2311.13870) | 本研究引入了一种多意图逆Q学习算法，用于解决在推断离散时变奖励时的挑战。通过聚类观察到的专家轨迹并独立解决每个意图的逆强化学习问题，我们的方法在动物行为预测方面超越了当前的基准，产生了可解释的奖励函数。 |
| [^2] | [Disentangling Hippocampal Shape Variations: A Study of Neurological Disorders Using Graph Variational Autoencoder with Contrastive Learning](https://arxiv.org/abs/2404.00785) | 本研究利用图变分自动编码器和对比学习解开神经系统疾病中海马形状变异的关键潜变量，超越了其他先进方法在解开能力上的表现。 |

# 详细

[^1]: 可解释行为表示的多意图逆Q学习

    Multi-intention Inverse Q-learning for Interpretable Behavior Representation

    [https://rss.arxiv.org/abs/2311.13870](https://rss.arxiv.org/abs/2311.13870)

    本研究引入了一种多意图逆Q学习算法，用于解决在推断离散时变奖励时的挑战。通过聚类观察到的专家轨迹并独立解决每个意图的逆强化学习问题，我们的方法在动物行为预测方面超越了当前的基准，产生了可解释的奖励函数。

    

    在推动决策过程理解方面，逆强化学习（IRL）在重构动物复杂行为中的多个意图方面证明了其重要性。鉴于最近发展的连续时间多意图IRL框架，人们一直在研究如何使用IRL推断离散的时变奖励。为了解决这个挑战，我们引入了潜（马尔科夫）变量逆Q学习（L(M)V-IQL），这是一种专门用于适应离散内在奖励函数的IRL算法。通过利用期望最大化方法，我们将观察到的专家轨迹聚类成不同的意图，并为每个意图独立解决IRL问题。通过模拟实验和对不同真实鼠类行为数据集的应用，我们的方法在动物行为预测方面超越了当前的基准，产生了可解释的奖励函数。这一进展有望打开推动科学与工程应用的新机遇。

    In advancing the understanding of decision-making processes, Inverse Reinforcement Learning (IRL) have proven instrumental in reconstructing animal's multiple intentions amidst complex behaviors. Given the recent development of a continuous-time multi-intention IRL framework, there has been persistent inquiry into inferring discrete time-varying rewards with IRL. To tackle the challenge, we introduce Latent (Markov) Variable Inverse Q-learning (L(M)V-IQL), a novel class of IRL algorthms tailored for accommodating discrete intrinsic reward functions. Leveraging an Expectation-Maximization approach, we cluster observed expert trajectories into distinct intentions and independently solve the IRL problem for each. Demonstrating the efficacy of L(M)V-IQL through simulated experiments and its application to different real mouse behavior datasets, our approach surpasses current benchmarks in animal behavior prediction, producing interpretable reward functions. This advancement holds promise f
    
[^2]: 解开海马形状变异之谜：利用对比学习的图变分自动编码器研究神经系统疾病

    Disentangling Hippocampal Shape Variations: A Study of Neurological Disorders Using Graph Variational Autoencoder with Contrastive Learning

    [https://arxiv.org/abs/2404.00785](https://arxiv.org/abs/2404.00785)

    本研究利用图变分自动编码器和对比学习解开神经系统疾病中海马形状变异的关键潜变量，超越了其他先进方法在解开能力上的表现。

    

    本文提出了一项综合研究，专注于在神经系统疾病背景下从扩散张量成像（DTI）数据集中解开海马形状变异。借助增强的监督对比学习图变分自动编码器（VAE），我们的方法旨在通过区分代表年龄和是否患病的两个不同潜变量来提高解释性。在我们的消融研究中，我们调查了一系列VAE架构和对比损失函数，展示了我们方法增强的解开能力。这个评估使用了来自DTI海马数据集的合成3D环形网格数据和真实的3D海马网格数据集。我们的监督解开模型在解开分数方面优于几种最先进的方法，如属性和引导VAE。我们的模型可以区分不同年龄组和疾病状况。

    arXiv:2404.00785v1 Announce Type: cross  Abstract: This paper presents a comprehensive study focused on disentangling hippocampal shape variations from diffusion tensor imaging (DTI) datasets within the context of neurological disorders. Leveraging a Graph Variational Autoencoder (VAE) enhanced with Supervised Contrastive Learning, our approach aims to improve interpretability by disentangling two distinct latent variables corresponding to age and the presence of diseases. In our ablation study, we investigate a range of VAE architectures and contrastive loss functions, showcasing the enhanced disentanglement capabilities of our approach. This evaluation uses synthetic 3D torus mesh data and real 3D hippocampal mesh datasets derived from the DTI hippocampal dataset. Our supervised disentanglement model outperforms several state-of-the-art (SOTA) methods like attribute and guided VAEs in terms of disentanglement scores. Our model distinguishes between age groups and disease status in pa
    

