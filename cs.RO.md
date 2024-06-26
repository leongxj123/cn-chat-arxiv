# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Safe Task Planning for Language-Instructed Multi-Robot Systems using Conformal Prediction](https://arxiv.org/abs/2402.15368) | 本文引入了一种新的基于分布式LLM和符合预测技术的多机器人规划器，实现了高任务成功率。 |
| [^2] | [Who Plays First? Optimizing the Order of Play in Stackelberg Games with Many Robots](https://arxiv.org/abs/2402.09246) | 本论文研究了在Stackelberg博弈中优化众多机器人的行动顺序的问题，并引入了一个高效准确的算法(B&P)来求解相关的优化问题和均衡。该算法具有广泛的实际应用。 |

# 详细

[^1]: 使用符合预测的技术实现语言指导多机器人系统的安全任务规划

    Safe Task Planning for Language-Instructed Multi-Robot Systems using Conformal Prediction

    [https://arxiv.org/abs/2402.15368](https://arxiv.org/abs/2402.15368)

    本文引入了一种新的基于分布式LLM和符合预测技术的多机器人规划器，实现了高任务成功率。

    

    本文解决了语言指导机器人团队的任务规划问题。任务用自然语言（NL）表示，要求机器人在各种位置和语义对象上应用它们的能力（例如移动、操作和感知）。最近几篇论文通过利用预训练的大型语言模型（LLMs）设计有效的多机器人计划来解决类似的规划问题。然而，这些方法缺乏任务性能和安全性保证。为了解决这一挑战，我们引入了一种新的基于分布式LLM的规划器，能够实现高任务成功率。这是通过利用符合预测（CP）来实现的，CP是一种基于分布的不确定性量化工具，可以在黑盒模型中对其固有不确定性进行推理。CP允许所提出的多机器人规划器以分布方式推理其固有不确定性，使得机器人在充分信任时能够做出个别决策。

    arXiv:2402.15368v1 Announce Type: cross  Abstract: This paper addresses task planning problems for language-instructed robot teams. Tasks are expressed in natural language (NL), requiring the robots to apply their capabilities (e.g., mobility, manipulation, and sensing) at various locations and semantic objects. Several recent works have addressed similar planning problems by leveraging pre-trained Large Language Models (LLMs) to design effective multi-robot plans. However, these approaches lack mission performance and safety guarantees. To address this challenge, we introduce a new decentralized LLM-based planner that is capable of achieving high mission success rates. This is accomplished by leveraging conformal prediction (CP), a distribution-free uncertainty quantification tool in black-box models. CP allows the proposed multi-robot planner to reason about its inherent uncertainty in a decentralized fashion, enabling robots to make individual decisions when they are sufficiently ce
    
[^2]: 谁先行动？优化Stackelberg博弈中众多机器人的行动顺序

    Who Plays First? Optimizing the Order of Play in Stackelberg Games with Many Robots

    [https://arxiv.org/abs/2402.09246](https://arxiv.org/abs/2402.09246)

    本论文研究了在Stackelberg博弈中优化众多机器人的行动顺序的问题，并引入了一个高效准确的算法(B&P)来求解相关的优化问题和均衡。该算法具有广泛的实际应用。

    

    我们考虑计算多智能体空间导航问题的社会最优行动顺序的问题，即智能体决策顺序，以及与之相关的N人Stackelberg轨迹博弈的均衡。我们将这个问题建模为一个混合整数优化问题，涉及到所有可能的行动顺序的Stackelberg博弈空间。为了解决这个问题，我们引入了Branch and Play (B&P)，这是一个高效且准确的算法，可以收敛到社会最优行动顺序及其Stackelberg均衡。作为B&P的一个子例程，我们提出并扩展了顺序轨迹规划，即一种流行的多智能体控制方法，以便为任何给定的行动顺序可扩展地计算有效的本地Stackelberg均衡。我们证明了B&P在协调空中交通控制、群体形成和交付车队方面的实际效用。我们发现B&P的结果是一致的。

    arXiv:2402.09246v1 Announce Type: cross Abstract: We consider the multi-agent spatial navigation problem of computing the socially optimal order of play, i.e., the sequence in which the agents commit to their decisions, and its associated equilibrium in an N-player Stackelberg trajectory game. We model this problem as a mixed-integer optimization problem over the space of all possible Stackelberg games associated with the order of play's permutations. To solve the problem, we introduce Branch and Play (B&P), an efficient and exact algorithm that provably converges to a socially optimal order of play and its Stackelberg equilibrium. As a subroutine for B&P, we employ and extend sequential trajectory planning, i.e., a popular multi-agent control approach, to scalably compute valid local Stackelberg equilibria for any given order of play. We demonstrate the practical utility of B&P to coordinate air traffic control, swarm formation, and delivery vehicle fleets. We find that B&P consistent
    

