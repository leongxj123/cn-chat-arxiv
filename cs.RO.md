# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Human-compatible driving partners through data-regularized self-play reinforcement learning](https://arxiv.org/abs/2403.19648) | 提出了Human-Regularized PPO (HR-PPO)算法，通过自我博弈训练代理，实现在封闭环境中逼真且有效的驾驶伙伴 |
| [^2] | [Who Plays First? Optimizing the Order of Play in Stackelberg Games with Many Robots](https://arxiv.org/abs/2402.09246) | 本论文研究了在Stackelberg博弈中优化众多机器人的行动顺序的问题，并引入了一个高效准确的算法(B&P)来求解相关的优化问题和均衡。该算法具有广泛的实际应用。 |

# 详细

[^1]: 通过数据正则化的自我博弈强化学习实现与人类兼容的驾驶伙伴

    Human-compatible driving partners through data-regularized self-play reinforcement learning

    [https://arxiv.org/abs/2403.19648](https://arxiv.org/abs/2403.19648)

    提出了Human-Regularized PPO (HR-PPO)算法，通过自我博弈训练代理，实现在封闭环境中逼真且有效的驾驶伙伴

    

    自主驾驶汽车面临的一个核心挑战是与人类进行协调。因此，在模拟环境中，将逼真的人类代理纳入自动驾驶系统的可扩展训练和评估是至关重要的。我们提出了一种名为Human-Regularized PPO (HR-PPO)的多智能体算法，其中代理通过自我博弈进行训练，对偏离人类参考策略的行为进行小幅惩罚，以构建在封闭环境中既逼真又有效的代理。

    arXiv:2403.19648v1 Announce Type: cross  Abstract: A central challenge for autonomous vehicles is coordinating with humans. Therefore, incorporating realistic human agents is essential for scalable training and evaluation of autonomous driving systems in simulation. Simulation agents are typically developed by imitating large-scale, high-quality datasets of human driving. However, pure imitation learning agents empirically have high collision rates when executed in a multi-agent closed-loop setting. To build agents that are realistic and effective in closed-loop settings, we propose Human-Regularized PPO (HR-PPO), a multi-agent algorithm where agents are trained through self-play with a small penalty for deviating from a human reference policy. In contrast to prior work, our approach is RL-first and only uses 30 minutes of imperfect human demonstrations. We evaluate agents in a large set of multi-agent traffic scenes. Results show our HR-PPO agents are highly effective in achieving goa
    
[^2]: 谁先行动？优化Stackelberg博弈中众多机器人的行动顺序

    Who Plays First? Optimizing the Order of Play in Stackelberg Games with Many Robots

    [https://arxiv.org/abs/2402.09246](https://arxiv.org/abs/2402.09246)

    本论文研究了在Stackelberg博弈中优化众多机器人的行动顺序的问题，并引入了一个高效准确的算法(B&P)来求解相关的优化问题和均衡。该算法具有广泛的实际应用。

    

    我们考虑计算多智能体空间导航问题的社会最优行动顺序的问题，即智能体决策顺序，以及与之相关的N人Stackelberg轨迹博弈的均衡。我们将这个问题建模为一个混合整数优化问题，涉及到所有可能的行动顺序的Stackelberg博弈空间。为了解决这个问题，我们引入了Branch and Play (B&P)，这是一个高效且准确的算法，可以收敛到社会最优行动顺序及其Stackelberg均衡。作为B&P的一个子例程，我们提出并扩展了顺序轨迹规划，即一种流行的多智能体控制方法，以便为任何给定的行动顺序可扩展地计算有效的本地Stackelberg均衡。我们证明了B&P在协调空中交通控制、群体形成和交付车队方面的实际效用。我们发现B&P的结果是一致的。

    arXiv:2402.09246v1 Announce Type: cross Abstract: We consider the multi-agent spatial navigation problem of computing the socially optimal order of play, i.e., the sequence in which the agents commit to their decisions, and its associated equilibrium in an N-player Stackelberg trajectory game. We model this problem as a mixed-integer optimization problem over the space of all possible Stackelberg games associated with the order of play's permutations. To solve the problem, we introduce Branch and Play (B&P), an efficient and exact algorithm that provably converges to a socially optimal order of play and its Stackelberg equilibrium. As a subroutine for B&P, we employ and extend sequential trajectory planning, i.e., a popular multi-agent control approach, to scalably compute valid local Stackelberg equilibria for any given order of play. We demonstrate the practical utility of B&P to coordinate air traffic control, swarm formation, and delivery vehicle fleets. We find that B&P consistent
    

