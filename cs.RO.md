# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RASP: A Drone-based Reconfigurable Actuation and Sensing Platform Towards Ambient Intelligent Systems](https://arxiv.org/abs/2403.12853) | 提出了RASP，一个可在25秒内自主更换传感器和执行器的模块化和可重构传感和作动平台，使无人机能快速适应各种任务，同时引入了利用大规模语言和视觉语言模型的个人助理系统架构。 |
| [^2] | [Introduction to Online Nonstochastic Control.](http://arxiv.org/abs/2211.09619) | 介绍了一种新兴的在线非随机控制方法，通过在一组策略中寻找低后悔，获得对最优策略的近似。 |

# 详细

[^1]: 基于无人机的环境智能系统的可重构作动和传感平台RASP

    RASP: A Drone-based Reconfigurable Actuation and Sensing Platform Towards Ambient Intelligent Systems

    [https://arxiv.org/abs/2403.12853](https://arxiv.org/abs/2403.12853)

    提出了RASP，一个可在25秒内自主更换传感器和执行器的模块化和可重构传感和作动平台，使无人机能快速适应各种任务，同时引入了利用大规模语言和视觉语言模型的个人助理系统架构。

    

    实现消费级无人机与我们家中的吸尘机器人或日常生活中的个人智能手机一样有用，需要无人机能感知、驱动和响应可能出现的一般情况。为了实现这一愿景，我们提出了RASP，一个模块化和可重构的传感和作动平台，允许无人机在仅25秒内自主更换机载传感器和执行器，使单个无人机能够快速适应各种任务。RASP包括一个机械层，用于物理更换传感器模块，一个电气层，用于维护传感器/执行器的电源和通信线路，以及一个软件层，用于在无人机和我们平台上的任何传感器模块之间维护一个公共接口。利用最近在大型语言和视觉语言模型方面的进展，我们进一步介绍了一种利用RASP的个人助理系统的架构、实现和现实世界部署。

    arXiv:2403.12853v1 Announce Type: cross  Abstract: Realizing consumer-grade drones that are as useful as robot vacuums throughout our homes or personal smartphones in our daily lives requires drones to sense, actuate, and respond to general scenarios that may arise. Towards this vision, we propose RASP, a modular and reconfigurable sensing and actuation platform that allows drones to autonomously swap onboard sensors and actuators in only 25 seconds, allowing a single drone to quickly adapt to a diverse range of tasks. RASP consists of a mechanical layer to physically swap sensor modules, an electrical layer to maintain power and communication lines to the sensor/actuator, and a software layer to maintain a common interface between the drone and any sensor module in our platform. Leveraging recent advances in large language and visual language models, we further introduce the architecture, implementation, and real-world deployments of a personal assistant system utilizing RASP. We demo
    
[^2]: 在线非随机控制简介

    Introduction to Online Nonstochastic Control. (arXiv:2211.09619v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.09619](http://arxiv.org/abs/2211.09619)

    介绍了一种新兴的在线非随机控制方法，通过在一组策略中寻找低后悔，获得对最优策略的近似。

    

    本文介绍了一种新兴的动态系统控制与可微强化学习范式——在线非随机控制，并应用在线凸优化和凸松弛技术得到了具有可证明保证的新方法，在最佳和鲁棒控制方面取得了显著成果。与其他框架不同，该方法的目标是对抗性攻击，在无法预测扰动模型的情况下，通过在一组策略中寻找低后悔，获得对最优策略的近似。

    This text presents an introduction to an emerging paradigm in control of dynamical systems and differentiable reinforcement learning called online nonstochastic control. The new approach applies techniques from online convex optimization and convex relaxations to obtain new methods with provable guarantees for classical settings in optimal and robust control.  The primary distinction between online nonstochastic control and other frameworks is the objective. In optimal control, robust control, and other control methodologies that assume stochastic noise, the goal is to perform comparably to an offline optimal strategy. In online nonstochastic control, both the cost functions as well as the perturbations from the assumed dynamical model are chosen by an adversary. Thus the optimal policy is not defined a priori. Rather, the target is to attain low regret against the best policy in hindsight from a benchmark class of policies.  This objective suggests the use of the decision making frame
    

