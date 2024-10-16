# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CtRL-Sim: Reactive and Controllable Driving Agents with Offline Reinforcement Learning](https://arxiv.org/abs/2403.19918) | CtRL-Sim提出了一种利用离线强化学习生成反应性和可控交通代理的方法，通过在Nocturne模拟器中处理真实世界的驾驶数据来实现这一目标。 |
| [^2] | [Learning Quadruped Locomotion Using Differentiable Simulation](https://arxiv.org/abs/2403.14864) | 本文提出了一种新的可微分仿真框架，通过将复杂的全身仿真解耦为两个单独的连续域，并与更精确的模型对齐，来克服四足动作中的不连续性挑战。 |

# 详细

[^1]: CtRL-Sim：使用离线强化学习的反应性可控驾驶代理

    CtRL-Sim: Reactive and Controllable Driving Agents with Offline Reinforcement Learning

    [https://arxiv.org/abs/2403.19918](https://arxiv.org/abs/2403.19918)

    CtRL-Sim提出了一种利用离线强化学习生成反应性和可控交通代理的方法，通过在Nocturne模拟器中处理真实世界的驾驶数据来实现这一目标。

    

    在这项工作中，我们提出了CtRL-Sim，一种利用物理增强的Nocturne模拟器中的回报条件化离线强化学习来高效生成反应性和可控交通代理的方法。具体来说，我们通过Nocturne模拟器处理真实世界的驾驶数据，以生成多样化的离线数据。

    arXiv:2403.19918v1 Announce Type: cross  Abstract: Evaluating autonomous vehicle stacks (AVs) in simulation typically involves replaying driving logs from real-world recorded traffic. However, agents replayed from offline data do not react to the actions of the AV, and their behaviour cannot be easily controlled to simulate counterfactual scenarios. Existing approaches have attempted to address these shortcomings by proposing methods that rely on heuristics or learned generative models of real-world data but these approaches either lack realism or necessitate costly iterative sampling procedures to control the generated behaviours. In this work, we take an alternative approach and propose CtRL-Sim, a method that leverages return-conditioned offline reinforcement learning within a physics-enhanced Nocturne simulator to efficiently generate reactive and controllable traffic agents. Specifically, we process real-world driving data through the Nocturne simulator to generate a diverse offli
    
[^2]: 使用可微分仿真学习四足动作

    Learning Quadruped Locomotion Using Differentiable Simulation

    [https://arxiv.org/abs/2403.14864](https://arxiv.org/abs/2403.14864)

    本文提出了一种新的可微分仿真框架，通过将复杂的全身仿真解耦为两个单独的连续域，并与更精确的模型对齐，来克服四足动作中的不连续性挑战。

    

    最近大部分机器人运动控制的进展都是由无模型强化学习驱动的，本文探讨了可微分仿真的潜力。可微分仿真通过使用机器人模型计算低变异一阶梯度，承诺了更快的收敛速度和更稳定的训练，但到目前为止，其在四足机器人控制方面的应用仍然有限。可微分仿真面临的主要挑战在于由于接触丰富环境（如四足动作）中的不连续性，导致机器人任务的复杂优化景观。本文提出了一个新的可微分仿真框架以克服这些挑战。关键想法包括将可能由于接触而出现不连续性的复杂全身仿真解耦为两个单独的连续域。随后，我们将简化模型产生的机器人状态与更精确的不可微分模型对齐。

    arXiv:2403.14864v1 Announce Type: cross  Abstract: While most recent advancements in legged robot control have been driven by model-free reinforcement learning, we explore the potential of differentiable simulation. Differentiable simulation promises faster convergence and more stable training by computing low-variant first-order gradients using the robot model, but so far, its use for legged robot control has remained limited to simulation. The main challenge with differentiable simulation lies in the complex optimization landscape of robotic tasks due to discontinuities in contact-rich environments, e.g., quadruped locomotion. This work proposes a new, differentiable simulation framework to overcome these challenges. The key idea involves decoupling the complex whole-body simulation, which may exhibit discontinuities due to contact, into two separate continuous domains. Subsequently, we align the robot state resulting from the simplified model with a more precise, non-differentiable 
    

