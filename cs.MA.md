# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Context-aware Communication for Multi-agent Reinforcement Learning.](http://arxiv.org/abs/2312.15600) | 这项研究针对多智能体强化学习提出了一种上下文感知的通信方案，通过两个阶段的交流，使智能体能够发送个性化的消息，从而提高合作和团队性能。 |

# 详细

[^1]: 多智能体强化学习的上下文感知通信

    Context-aware Communication for Multi-agent Reinforcement Learning. (arXiv:2312.15600v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2312.15600](http://arxiv.org/abs/2312.15600)

    这项研究针对多智能体强化学习提出了一种上下文感知的通信方案，通过两个阶段的交流，使智能体能够发送个性化的消息，从而提高合作和团队性能。

    

    对于多智能体强化学习（MARL），有效的通信协议对于促进合作和提高团队性能至关重要。为了利用通信，许多以前的工作提出将本地信息压缩成一条消息并广播给所有可达的智能体。然而，这种简单的消息传递机制可能无法为个体智能体提供足够、关键和相关的信息，特别是在带宽严重有限的场景下。这激励我们为MARL开发上下文感知的通信方案，旨在向不同的智能体发送个性化的消息。我们的通信协议名为CACOM，由两个阶段组成。第一个阶段中，智能体以广播方式交换粗略表示，为第二个阶段提供上下文信息。紧随其后，智能体在第二个阶段中利用注意机制为接收者选择性生成个性化的消息。此外，我们还采用了学习的步长量化方法。

    Effective communication protocols in multi-agent reinforcement learning (MARL) are critical to fostering cooperation and enhancing team performance. To leverage communication, many previous works have proposed to compress local information into a single message and broadcast it to all reachable agents. This simplistic messaging mechanism, however, may fail to provide adequate, critical, and relevant information to individual agents, especially in severely bandwidth-limited scenarios. This motivates us to develop context-aware communication schemes for MARL, aiming to deliver personalized messages to different agents. Our communication protocol, named CACOM, consists of two stages. In the first stage, agents exchange coarse representations in a broadcast fashion, providing context for the second stage. Following this, agents utilize attention mechanisms in the second stage to selectively generate messages personalized for the receivers. Furthermore, we employ the learned step size quant
    

