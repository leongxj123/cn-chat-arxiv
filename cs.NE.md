# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diffusion Policies for Out-of-Distribution Generalization in Offline Reinforcement Learning.](http://arxiv.org/abs/2307.04726) | 该论文介绍了一种名为状态重构扩散策略 (SRDP) 的新方法，该方法在最新的扩散策略类中引入了状态重构特征学习，以解决脱机强化学习中的分布偏移和有效表示策略的问题。 |

# 详细

[^1]: 脱机强化学习中的离散策略的扩散策略

    Diffusion Policies for Out-of-Distribution Generalization in Offline Reinforcement Learning. (arXiv:2307.04726v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2307.04726](http://arxiv.org/abs/2307.04726)

    该论文介绍了一种名为状态重构扩散策略 (SRDP) 的新方法，该方法在最新的扩散策略类中引入了状态重构特征学习，以解决脱机强化学习中的分布偏移和有效表示策略的问题。

    

    脱机强化学习 (RL) 方法利用以前的经验来学习比用于数据收集的行为策略更好的策略。与行为克隆相反，行为克隆假设数据是从专家演示中收集的，而脱机 RL 可以使用非专家数据和多模态行为策略。然而，脱机 RL 算法在处理分布偏移和有效表示策略方面面临挑战，因为训练过程中缺乏在线交互。先前关于脱机 RL 的工作使用条件扩散模型来表示数据集中的多模态行为。然而，这些方法并没有针对缓解脱机分布状态泛化而制定。我们介绍了一种新的方法，名为状态重构扩散策略 (SRDP)，将状态重构特征学习纳入到最新的扩散策略类中，以解决脱机分布通用化问题。状态重构损失促进了更详细的描述。

    Offline Reinforcement Learning (RL) methods leverage previous experiences to learn better policies than the behavior policy used for data collection. In contrast to behavior cloning, which assumes the data is collected from expert demonstrations, offline RL can work with non-expert data and multimodal behavior policies. However, offline RL algorithms face challenges in handling distribution shifts and effectively representing policies due to the lack of online interaction during training. Prior work on offline RL uses conditional diffusion models to represent multimodal behavior in the dataset. Nevertheless, these methods are not tailored toward alleviating the out-of-distribution state generalization. We introduce a novel method, named State Reconstruction for Diffusion Policies (SRDP), incorporating state reconstruction feature learning in the recent class of diffusion policies to address the out-of-distribution generalization problem. State reconstruction loss promotes more descript
    

