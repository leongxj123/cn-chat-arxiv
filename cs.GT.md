# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bandits with Deterministically Evolving States.](http://arxiv.org/abs/2307.11655) | 该论文提出了一种名为具有确定性演化状态的强盗模型，用于学习带有强盗反馈的推荐系统和在线广告。该模型考虑了状态演化的不同速率，能准确评估奖励与系统健康程度之间的关系。 |

# 详细

[^1]: 具有确定性演化状态的强盗模型

    Bandits with Deterministically Evolving States. (arXiv:2307.11655v1 [cs.LG])

    [http://arxiv.org/abs/2307.11655](http://arxiv.org/abs/2307.11655)

    该论文提出了一种名为具有确定性演化状态的强盗模型，用于学习带有强盗反馈的推荐系统和在线广告。该模型考虑了状态演化的不同速率，能准确评估奖励与系统健康程度之间的关系。

    

    我们提出了一种学习与强盗反馈结合的模型，同时考虑到确定性演化和不可观测的状态，我们称之为具有确定性演化状态的强盗模型。我们的模型主要应用于推荐系统和在线广告的学习。在这两种情况下，算法在每一轮获得的奖励是选择行动的短期奖励和系统的“健康”程度（即通过其状态测量）的函数。例如，在推荐系统中，平台从用户对特定类型内容的参与中获得的奖励不仅取决于具体内容的固有特征，还取决于用户与平台上其他类型内容互动后其偏好的演化。我们的通用模型考虑了状态演化的不同速率λ∈[0,1]（例如，用户的偏好因先前内容消费而快速变化）。

    We propose a model for learning with bandit feedback while accounting for deterministically evolving and unobservable states that we call Bandits with Deterministically Evolving States. The workhorse applications of our model are learning for recommendation systems and learning for online ads. In both cases, the reward that the algorithm obtains at each round is a function of the short-term reward of the action chosen and how ``healthy'' the system is (i.e., as measured by its state). For example, in recommendation systems, the reward that the platform obtains from a user's engagement with a particular type of content depends not only on the inherent features of the specific content, but also on how the user's preferences have evolved as a result of interacting with other types of content on the platform. Our general model accounts for the different rate $\lambda \in [0,1]$ at which the state evolves (e.g., how fast a user's preferences shift as a result of previous content consumption
    

