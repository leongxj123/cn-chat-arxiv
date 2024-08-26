# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A PPO-based DRL Auto-Tuning Nonlinear PID Drone Controller for Robust Autonomous Flights](https://arxiv.org/abs/2404.00204) | 该项目将非线性深度强化学习（DRL）代理引入无人机控制中，取代传统线性PID控制器，实现了无缝过渡、提高响应速度和稳定性，同时结合PPO策略训练DRL代理，并利用高精度跟踪系统提高自主飞行精度。 |
| [^2] | [Intelligent Energy Management with IoT Framework in Smart Cities Using Intelligent Analysis: An Application of Machine Learning Methods for Complex Networks and Systems.](http://arxiv.org/abs/2306.05567) | 本研究开发了一个智能城市能源管理的物联网框架，结合智能分析和多组件的架构，研究了基于智能机制的智能能源管理解决方案，以期节能和优化管理。 |

# 详细

[^1]: 基于PPO的DRL自调PID非线性无人机控制器用于稳健自主飞行

    A PPO-based DRL Auto-Tuning Nonlinear PID Drone Controller for Robust Autonomous Flights

    [https://arxiv.org/abs/2404.00204](https://arxiv.org/abs/2404.00204)

    该项目将非线性深度强化学习（DRL）代理引入无人机控制中，取代传统线性PID控制器，实现了无缝过渡、提高响应速度和稳定性，同时结合PPO策略训练DRL代理，并利用高精度跟踪系统提高自主飞行精度。

    

    该项目旨在通过将非线性深度强化学习（DRL）代理作为传统线性比例积分微分（PID）控制器的替代品，从而彻底改变无人机飞行控制。主要目标是在手动和自主模式之间实现无缝过渡，提高响应速度和稳定性。我们在Gazebo模拟器中利用近端策略优化（PPO）强化学习策略来训练DRL代理。添加20000美元的室内Vicon跟踪系统提供<1mm的定位精度，显着提高了自主飞行精度。为了在最短的无碰撞轨迹中导航无人机，我们还建立了一个三维A*路径规划器并成功地将其实施到实际飞行中。

    arXiv:2404.00204v1 Announce Type: cross  Abstract: This project aims to revolutionize drone flight control by implementing a nonlinear Deep Reinforcement Learning (DRL) agent as a replacement for traditional linear Proportional Integral Derivative (PID) controllers. The primary objective is to seamlessly transition drones between manual and autonomous modes, enhancing responsiveness and stability. We utilize the Proximal Policy Optimization (PPO) reinforcement learning strategy within the Gazebo simulator to train the DRL agent. Adding a $20,000 indoor Vicon tracking system offers <1mm positioning accuracy, which significantly improves autonomous flight precision. To navigate the drone in the shortest collision-free trajectory, we also build a 3 dimensional A* path planner and implement it into the real flight successfully.
    
[^2]: 智能分析，在物联网框架下的智能城市能源管理：复杂网络和系统机器学习方法应用的案例研究

    Intelligent Energy Management with IoT Framework in Smart Cities Using Intelligent Analysis: An Application of Machine Learning Methods for Complex Networks and Systems. (arXiv:2306.05567v1 [cs.LG])

    [http://arxiv.org/abs/2306.05567](http://arxiv.org/abs/2306.05567)

    本研究开发了一个智能城市能源管理的物联网框架，结合智能分析和多组件的架构，研究了基于智能机制的智能能源管理解决方案，以期节能和优化管理。

    

    智能建筑越来越多地使用基于物联网的无线传感系统来降低能源消耗和环境影响。本研究的主要贡献是开发了一个全面的基于物联网的智能城市能源管理框架，融合了多个物联网架构和框架的组件。该框架通过智能分析，不仅收集和存储信息，而且还是其他企业开发应用的平台。此外，我们还研究了基于智能机制的智能能源管理解决方案。能源资源的消耗和需求增加导致了节能与优化管理的需求和挑战。

    Smart buildings are increasingly using Internet of Things (IoT)-based wireless sensing systems to reduce their energy consumption and environmental impact. As a result of their compact size and ability to sense, measure, and compute all electrical properties, Internet of Things devices have become increasingly important in our society. A major contribution of this study is the development of a comprehensive IoT-based framework for smart city energy management, incorporating multiple components of IoT architecture and framework. An IoT framework for intelligent energy management applications that employ intelligent analysis is an essential system component that collects and stores information. Additionally, it serves as a platform for the development of applications by other companies. Furthermore, we have studied intelligent energy management solutions based on intelligent mechanisms. The depletion of energy resources and the increase in energy demand have led to an increase in energy 
    

