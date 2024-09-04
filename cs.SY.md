# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A PPO-based DRL Auto-Tuning Nonlinear PID Drone Controller for Robust Autonomous Flights](https://arxiv.org/abs/2404.00204) | 该项目将非线性深度强化学习（DRL）代理引入无人机控制中，取代传统线性PID控制器，实现了无缝过渡、提高响应速度和稳定性，同时结合PPO策略训练DRL代理，并利用高精度跟踪系统提高自主飞行精度。 |
| [^2] | [An experimental evaluation of Deep Reinforcement Learning algorithms for HVAC control.](http://arxiv.org/abs/2401.05737) | 本论文通过对HVAC控制的几种最先进的深度强化学习算法进行了实验评估，发现SAC和TD3等算法在复杂场景中具有潜力，并揭示了与泛化和增量学习相关的挑战。 |
| [^3] | [Autonomous Payload Thermal Control.](http://arxiv.org/abs/2307.15438) | 该论文提出了一种基于深度强化学习的框架，利用软演员-评论家算法在卫星上学习热控制策略，以解决小型卫星中热控制的挑战。该框架在模拟环境和实际空间处理计算机上进行了评估，并证明能够辅助传统热控制系统，保持载荷温度在可操作范围内。 |
| [^4] | [Active Learning of Discrete-Time Dynamics for Uncertainty-Aware Model Predictive Control.](http://arxiv.org/abs/2210.12583) | 本文提出了一种用于主动学习非线性机器人系统动力学的方法，结合了离线和在线学习，能够在实时中准确推断模型动力学，并设计了一种不确定性感知模型预测控制器。 |

# 详细

[^1]: 基于PPO的DRL自调PID非线性无人机控制器用于稳健自主飞行

    A PPO-based DRL Auto-Tuning Nonlinear PID Drone Controller for Robust Autonomous Flights

    [https://arxiv.org/abs/2404.00204](https://arxiv.org/abs/2404.00204)

    该项目将非线性深度强化学习（DRL）代理引入无人机控制中，取代传统线性PID控制器，实现了无缝过渡、提高响应速度和稳定性，同时结合PPO策略训练DRL代理，并利用高精度跟踪系统提高自主飞行精度。

    

    该项目旨在通过将非线性深度强化学习（DRL）代理作为传统线性比例积分微分（PID）控制器的替代品，从而彻底改变无人机飞行控制。主要目标是在手动和自主模式之间实现无缝过渡，提高响应速度和稳定性。我们在Gazebo模拟器中利用近端策略优化（PPO）强化学习策略来训练DRL代理。添加20000美元的室内Vicon跟踪系统提供<1mm的定位精度，显着提高了自主飞行精度。为了在最短的无碰撞轨迹中导航无人机，我们还建立了一个三维A*路径规划器并成功地将其实施到实际飞行中。

    arXiv:2404.00204v1 Announce Type: cross  Abstract: This project aims to revolutionize drone flight control by implementing a nonlinear Deep Reinforcement Learning (DRL) agent as a replacement for traditional linear Proportional Integral Derivative (PID) controllers. The primary objective is to seamlessly transition drones between manual and autonomous modes, enhancing responsiveness and stability. We utilize the Proximal Policy Optimization (PPO) reinforcement learning strategy within the Gazebo simulator to train the DRL agent. Adding a $20,000 indoor Vicon tracking system offers <1mm positioning accuracy, which significantly improves autonomous flight precision. To navigate the drone in the shortest collision-free trajectory, we also build a 3 dimensional A* path planner and implement it into the real flight successfully.
    
[^2]: HVAC控制的深度强化学习算法的实验评估

    An experimental evaluation of Deep Reinforcement Learning algorithms for HVAC control. (arXiv:2401.05737v1 [cs.LG])

    [http://arxiv.org/abs/2401.05737](http://arxiv.org/abs/2401.05737)

    本论文通过对HVAC控制的几种最先进的深度强化学习算法进行了实验评估，发现SAC和TD3等算法在复杂场景中具有潜力，并揭示了与泛化和增量学习相关的挑战。

    

    暖通空调系统是商业和居住建筑能源消耗的重要驱动因素。最近的研究表明，深度强化学习算法可以胜过传统的反应式控制器。然而，基于深度强化学习的解决方案通常是为特定设置而设计的，并且缺乏可比性的标准。为了填补这一空白，本文采用Sinergym框架，以舒适度和能源消耗为评判标准，对几种最先进的深度强化学习算法在HVAC控制方面进行了关键和可重现的评估。研究通过检查控制器的鲁棒性、适应性和优化目标之间的权衡，确认了SAC和TD3等深度强化学习算法在复杂场景中的潜力，并揭示了与泛化和增量学习相关的几个挑战。

    Heating, Ventilation, and Air Conditioning (HVAC) systems are a major driver of energy consumption in commercial and residential buildings. Recent studies have shown that Deep Reinforcement Learning (DRL) algorithms can outperform traditional reactive controllers. However, DRL-based solutions are generally designed for ad hoc setups and lack standardization for comparison. To fill this gap, this paper provides a critical and reproducible evaluation, in terms of comfort and energy consumption, of several state-of-the-art DRL algorithms for HVAC control. The study examines the controllers' robustness, adaptability, and trade-off between optimization goals by using the Sinergym framework. The results obtained confirm the potential of DRL algorithms, such as SAC and TD3, in complex scenarios and reveal several challenges related to generalization and incremental learning.
    
[^3]: 自主载荷热控制

    Autonomous Payload Thermal Control. (arXiv:2307.15438v1 [cs.LG])

    [http://arxiv.org/abs/2307.15438](http://arxiv.org/abs/2307.15438)

    该论文提出了一种基于深度强化学习的框架，利用软演员-评论家算法在卫星上学习热控制策略，以解决小型卫星中热控制的挑战。该框架在模拟环境和实际空间处理计算机上进行了评估，并证明能够辅助传统热控制系统，保持载荷温度在可操作范围内。

    

    在小型卫星中，热控制设备、科学仪器和电子部件的空间较小。此外，电子设备的近距离使得功耗散热困难，存在无法适当控制温度、降低部件寿命和任务性能的风险。为了应对这一挑战，利用卫星上逐渐增加的智能，提出了一种基于深度强化学习的框架，使用软演员-评论家算法来学习机载热控制策略。该框架在一个简单的模拟环境和未来将运往ISS并在IMAGIN-e任务中进行边缘计算的真实空间处理计算机中进行了评估。实验结果表明，所提出的框架能够学习控制载荷处理功率，以保持温度在操作范围内，补充传统热控制系统。

    In small satellites there is less room for heat control equipment, scientific instruments, and electronic components. Furthermore, the near proximity of the electronics makes power dissipation difficult, with the risk of not being able to control the temperature appropriately, reducing component lifetime and mission performance. To address this challenge, taking advantage of the advent of increasing intelligence on board satellites, a deep reinforcement learning based framework that uses Soft Actor-Critic algorithm is proposed for learning the thermal control policy onboard. The framework is evaluated both in a naive simulated environment and in a real space edge processing computer that will be shipped in the future IMAGIN-e mission and hosted in the ISS. The experiment results show that the proposed framework is able to learn to control the payload processing power to maintain the temperature under operational ranges, complementing traditional thermal control systems.
    
[^4]: 用于不确定性感知模型预测控制的离散时间动力学的主动学习

    Active Learning of Discrete-Time Dynamics for Uncertainty-Aware Model Predictive Control. (arXiv:2210.12583v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2210.12583](http://arxiv.org/abs/2210.12583)

    本文提出了一种用于主动学习非线性机器人系统动力学的方法，结合了离线和在线学习，能够在实时中准确推断模型动力学，并设计了一种不确定性感知模型预测控制器。

    

    模型驱动的控制需要对系统动力学进行准确建模，以便在复杂和动态环境中精确且安全地控制机器人。此外，在操作条件变化的情况下，模型应该不断调整以弥补动力学变化。本文提出了一种主动学习方法来主动建模非线性机器人系统的动力学。我们结合了离线学习以往经验和在线学习当前机器人与未知环境的交互。这两个因素使得学习过程高效且自适应，能够在实时中准确推断模型动力学，即使在大大不同于训练分布的操作范围内也可行。此外，我们设计了一种对学习到的动力学的aleatoric（数据）不确定性启发式条件的不确定性感知模型预测控制器。该控制器可以主动选择最优的控制动作。

    Model-based control requires an accurate model of the system dynamics for precisely and safely controlling the robot in complex and dynamic environments. Moreover, in the presence of variations in the operating conditions, the model should be continuously refined to compensate for dynamics changes. In this paper, we present a self-supervised learning approach that actively models the dynamics of nonlinear robotic systems. We combine offline learning from past experience and online learning from current robot interaction with the unknown environment. These two ingredients enable a highly sample-efficient and adaptive learning process, capable of accurately inferring model dynamics in real-time even in operating regimes that greatly differ from the training distribution. Moreover, we design an uncertainty-aware model predictive controller that is heuristically conditioned to the aleatoric (data) uncertainty of the learned dynamics. This controller actively chooses the optimal control act
    

