# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reinforcement Learning with Latent State Inference for Autonomous On-ramp Merging under Observation Delay](https://arxiv.org/abs/2403.11852) | 本文提出了一种具有潜在状态推断的强化学习方法，用于解决自动匝道合并问题，在没有详细了解周围车辆意图或驾驶风格的情况下安全执行匝道合并任务，并考虑了观测延迟，以增强代理在动态交通状况中的决策能力。 |
| [^2] | [DSSE: a drone swarm search environment.](http://arxiv.org/abs/2307.06240) | DSSE是一个无人机群集搜索环境，用于研究需要动态概率作为输入的强化学习算法。 |

# 详细

[^1]: 具有潜在状态推断的强化学习在自动匝道合并中的应用

    Reinforcement Learning with Latent State Inference for Autonomous On-ramp Merging under Observation Delay

    [https://arxiv.org/abs/2403.11852](https://arxiv.org/abs/2403.11852)

    本文提出了一种具有潜在状态推断的强化学习方法，用于解决自动匝道合并问题，在没有详细了解周围车辆意图或驾驶风格的情况下安全执行匝道合并任务，并考虑了观测延迟，以增强代理在动态交通状况中的决策能力。

    

    本文提出了一种解决自动匝道合并问题的新方法，其中自动驾驶车辆需要无缝地融入多车道高速公路上的车流。我们介绍了Lane-keeping, Lane-changing with Latent-state Inference and Safety Controller (L3IS)代理，旨在在没有关于周围车辆意图或驾驶风格的全面知识的情况下安全执行匝道合并任务。我们还提出了该代理的增强版AL3IS，考虑了观测延迟，使代理能够在具有车辆间通信延迟的现实环境中做出更稳健的决策。通过通过潜在状态建模环境中的不可观察方面，如其他驾驶员的意图，我们的方法增强了代理适应动态交通状况、优化合并操作并确保与其他车辆进行安全互动的能力。

    arXiv:2403.11852v1 Announce Type: cross  Abstract: This paper presents a novel approach to address the challenging problem of autonomous on-ramp merging, where a self-driving vehicle needs to seamlessly integrate into a flow of vehicles on a multi-lane highway. We introduce the Lane-keeping, Lane-changing with Latent-state Inference and Safety Controller (L3IS) agent, designed to perform the on-ramp merging task safely without comprehensive knowledge about surrounding vehicles' intents or driving styles. We also present an augmentation of this agent called AL3IS that accounts for observation delays, allowing the agent to make more robust decisions in real-world environments with vehicle-to-vehicle (V2V) communication delays. By modeling the unobservable aspects of the environment through latent states, such as other drivers' intents, our approach enhances the agent's ability to adapt to dynamic traffic conditions, optimize merging maneuvers, and ensure safe interactions with other vehi
    
[^2]: DSSE: 无人机群集搜索环境

    DSSE: a drone swarm search environment. (arXiv:2307.06240v1 [cs.LG])

    [http://arxiv.org/abs/2307.06240](http://arxiv.org/abs/2307.06240)

    DSSE是一个无人机群集搜索环境，用于研究需要动态概率作为输入的强化学习算法。

    

    无人机群集搜索项目是一个基于PettingZoo的环境，与多智能体（或单智能体）强化学习算法配合使用。该环境中的智能体（无人机）必须找到目标（遇险人员），但不知道目标的位置，并且不会根据自身与目标的距离得到奖励。但是，智能体会接收到目标出现在地图某个单元格的概率。该项目的目标是帮助研究需要动态概率作为输入的强化学习算法。

    The Drone Swarm Search project is an environment, based on PettingZoo, that is to be used in conjunction with multi-agent (or single-agent) reinforcement learning algorithms. It is an environment in which the agents (drones), have to find the targets (shipwrecked people). The agents do not know the position of the target and do not receive rewards related to their own distance to the target(s). However, the agents receive the probabilities of the target(s) being in a certain cell of the map. The aim of this project is to aid in the study of reinforcement learning algorithms that require dynamic probabilities as inputs.
    

