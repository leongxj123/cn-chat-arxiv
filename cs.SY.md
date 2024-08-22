# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ComTraQ-MPC: Meta-Trained DQN-MPC Integration for Trajectory Tracking with Limited Active Localization Updates](https://arxiv.org/abs/2403.01564) | ComTraQ-MPC是一个结合了DQN和MPC的新框架，旨在优化在有限主动定位更新下的轨迹跟踪。 |
| [^2] | [Suppressing unknown disturbances to dynamical systems using machine learning.](http://arxiv.org/abs/2307.03690) | 本文提出了一种使用机器学习的无模型方法，可以仅通过系统在已知强迫函数影响下的观测，识别和抑制未知系统的未知干扰。这项方法对训练函数有非常温和的限制，能够稳健地识别和抑制大类别的未知干扰。 |

# 详细

[^1]: ComTraQ-MPC：元训练的DQN-MPC集成用于具有有限主动定位更新的轨迹跟踪

    ComTraQ-MPC: Meta-Trained DQN-MPC Integration for Trajectory Tracking with Limited Active Localization Updates

    [https://arxiv.org/abs/2403.01564](https://arxiv.org/abs/2403.01564)

    ComTraQ-MPC是一个结合了DQN和MPC的新框架，旨在优化在有限主动定位更新下的轨迹跟踪。

    

    在局部可观察、随机环境中进行轨迹跟踪的最佳决策往往面临着一个重要挑战，即主动定位更新数量有限，这是指代理从传感器获取真实状态信息的过程。传统方法往往难以平衡资源保存、准确状态估计和精确跟踪之间的关系，导致性能次优。本文介绍了ComTraQ-MPC，这是一个结合了Deep Q-Networks (DQN)和模型预测控制(MPC)的新颖框架，旨在优化有限主动定位更新下的轨迹跟踪。元训练的DQN确保了自适应主动定位调度，同时

    arXiv:2403.01564v1 Announce Type: cross  Abstract: Optimal decision-making for trajectory tracking in partially observable, stochastic environments where the number of active localization updates -- the process by which the agent obtains its true state information from the sensors -- are limited, presents a significant challenge. Traditional methods often struggle to balance resource conservation, accurate state estimation and precise tracking, resulting in suboptimal performance. This problem is particularly pronounced in environments with large action spaces, where the need for frequent, accurate state data is paramount, yet the capacity for active localization updates is restricted by external limitations. This paper introduces ComTraQ-MPC, a novel framework that combines Deep Q-Networks (DQN) and Model Predictive Control (MPC) to optimize trajectory tracking with constrained active localization updates. The meta-trained DQN ensures adaptive active localization scheduling, while the
    
[^2]: 使用机器学习抑制动力系统中的未知干扰

    Suppressing unknown disturbances to dynamical systems using machine learning. (arXiv:2307.03690v1 [eess.SY])

    [http://arxiv.org/abs/2307.03690](http://arxiv.org/abs/2307.03690)

    本文提出了一种使用机器学习的无模型方法，可以仅通过系统在已知强迫函数影响下的观测，识别和抑制未知系统的未知干扰。这项方法对训练函数有非常温和的限制，能够稳健地识别和抑制大类别的未知干扰。

    

    识别和抑制动力系统中的未知干扰是一个在许多不同领域中应用的问题。在本文中，我们提出了一种无模型的方法，仅基于系统在已知强迫函数影响下的先前观测来识别和抑制未知系统的未知干扰。我们发现，在对训练函数有非常温和的限制下，我们的方法能够稳健地识别和抑制大类别的未知干扰。我们通过一个示例说明了我们的方案，其中识别和抑制了 Lorenz 系统的混沌干扰。

    Identifying and suppressing unknown disturbances to dynamical systems is a problem with applications in many different fields. In this Letter, we present a model-free method to identify and suppress an unknown disturbance to an unknown system based only on previous observations of the system under the influence of a known forcing function. We find that, under very mild restrictions on the training function, our method is able to robustly identify and suppress a large class of unknown disturbances. We illustrate our scheme with an example where a chaotic disturbance to the Lorenz system is identified and suppressed.
    

