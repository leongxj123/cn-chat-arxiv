# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Crowd-PrefRL: Preference-Based Reward Learning from Crowds.](http://arxiv.org/abs/2401.10941) | Crowd-PrefRL是一种基于众包的偏好反馈学习框架，能够从来自群体的反馈中学习奖励函数，并且能够强大地聚合群体偏好反馈并估计用户的可靠性。 |

# 详细

[^1]: Crowd-PrefRL: 基于众包的偏好反馈学习

    Crowd-PrefRL: Preference-Based Reward Learning from Crowds. (arXiv:2401.10941v1 [cs.HC])

    [http://arxiv.org/abs/2401.10941](http://arxiv.org/abs/2401.10941)

    Crowd-PrefRL是一种基于众包的偏好反馈学习框架，能够从来自群体的反馈中学习奖励函数，并且能够强大地聚合群体偏好反馈并估计用户的可靠性。

    

    基于偏好的强化学习提供了一个框架，通过对行为对的偏好进行人类反馈来训练智能体，使其能够在难以指定数值奖励函数的情况下学习期望的行为。尽管这个范式利用了人类的反馈，但目前将反馈视为单个人类用户所给出的。与此同时，以强大的方式合并来自群体（即用户集合）的偏好反馈仍然是一个挑战，而使用来自多个用户的反馈来训练强化学习智能体的问题仍然被研究不足。在这项工作中，我们引入了Crowd-PrefRL，一个利用来自群体的反馈进行基于偏好的强化学习的框架。这项工作展示了利用未知专业水平和可靠性的群体偏好反馈来学习奖励函数的可行性。Crowd-PrefRL不仅能够强大地聚合群体偏好反馈，还能够估计每个用户的可靠性。

    Preference-based reinforcement learning (RL) provides a framework to train agents using human feedback through pairwise preferences over pairs of behaviors, enabling agents to learn desired behaviors when it is difficult to specify a numerical reward function. While this paradigm leverages human feedback, it currently treats the feedback as given by a single human user. Meanwhile, incorporating preference feedback from crowds (i.e. ensembles of users) in a robust manner remains a challenge, and the problem of training RL agents using feedback from multiple human users remains understudied. In this work, we introduce Crowd-PrefRL, a framework for performing preference-based RL leveraging feedback from crowds. This work demonstrates the viability of learning reward functions from preference feedback provided by crowds of unknown expertise and reliability. Crowd-PrefRL not only robustly aggregates the crowd preference feedback, but also estimates the reliability of each user within the cr
    

