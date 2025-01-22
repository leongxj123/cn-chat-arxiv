# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A PPO-based DRL Auto-Tuning Nonlinear PID Drone Controller for Robust Autonomous Flights](https://arxiv.org/abs/2404.00204) | 该项目将非线性深度强化学习（DRL）代理引入无人机控制中，取代传统线性PID控制器，实现了无缝过渡、提高响应速度和稳定性，同时结合PPO策略训练DRL代理，并利用高精度跟踪系统提高自主飞行精度。 |
| [^2] | [Multi-Agent Consensus Seeking via Large Language Models.](http://arxiv.org/abs/2310.20151) | 本文研究了基于大型语言模型的多智能体系统中的一致性寻求问题。研究发现，在没有明确指导的情况下，智能体主要使用平均策略进行一致性寻求，同时还分析了智能体数量、智能体个性和网络拓扑对协商过程的影响。 |

# 详细

[^1]: 基于PPO的DRL自调PID非线性无人机控制器用于稳健自主飞行

    A PPO-based DRL Auto-Tuning Nonlinear PID Drone Controller for Robust Autonomous Flights

    [https://arxiv.org/abs/2404.00204](https://arxiv.org/abs/2404.00204)

    该项目将非线性深度强化学习（DRL）代理引入无人机控制中，取代传统线性PID控制器，实现了无缝过渡、提高响应速度和稳定性，同时结合PPO策略训练DRL代理，并利用高精度跟踪系统提高自主飞行精度。

    

    该项目旨在通过将非线性深度强化学习（DRL）代理作为传统线性比例积分微分（PID）控制器的替代品，从而彻底改变无人机飞行控制。主要目标是在手动和自主模式之间实现无缝过渡，提高响应速度和稳定性。我们在Gazebo模拟器中利用近端策略优化（PPO）强化学习策略来训练DRL代理。添加20000美元的室内Vicon跟踪系统提供<1mm的定位精度，显着提高了自主飞行精度。为了在最短的无碰撞轨迹中导航无人机，我们还建立了一个三维A*路径规划器并成功地将其实施到实际飞行中。

    arXiv:2404.00204v1 Announce Type: cross  Abstract: This project aims to revolutionize drone flight control by implementing a nonlinear Deep Reinforcement Learning (DRL) agent as a replacement for traditional linear Proportional Integral Derivative (PID) controllers. The primary objective is to seamlessly transition drones between manual and autonomous modes, enhancing responsiveness and stability. We utilize the Proximal Policy Optimization (PPO) reinforcement learning strategy within the Gazebo simulator to train the DRL agent. Adding a $20,000 indoor Vicon tracking system offers <1mm positioning accuracy, which significantly improves autonomous flight precision. To navigate the drone in the shortest collision-free trajectory, we also build a 3 dimensional A* path planner and implement it into the real flight successfully.
    
[^2]: 基于大型语言模型的多智能体一致性寻求

    Multi-Agent Consensus Seeking via Large Language Models. (arXiv:2310.20151v1 [cs.CL])

    [http://arxiv.org/abs/2310.20151](http://arxiv.org/abs/2310.20151)

    本文研究了基于大型语言模型的多智能体系统中的一致性寻求问题。研究发现，在没有明确指导的情况下，智能体主要使用平均策略进行一致性寻求，同时还分析了智能体数量、智能体个性和网络拓扑对协商过程的影响。

    

    大型语言模型（LLM）驱动的多智能体系统在协作解决复杂任务方面展现出了令人期待的能力。本研究考虑了多智能体协作中的一个基本问题：一致性寻求。当多个智能体一起工作时，我们关注的是它们如何通过智能体间的协商达成一致。为此，本研究研究了一个一致性寻求任务，其中每个智能体的状态是一个数值，并且它们通过相互协商来达成一致值。研究发现，当没有明确指导应采用哪种策略时，LLM驱动的智能体主要使用平均策略进行一致性寻求，尽管它们可能偶尔会使用其他策略。此外，本研究还分析了智能体数量、智能体个性和网络拓扑对协商过程的影响。本研究的发现有望为理解LLM驱动的多智能体行为奠定基础。

    Multi-agent systems driven by large language models (LLMs) have shown promising abilities for solving complex tasks in a collaborative manner. This work considers a fundamental problem in multi-agent collaboration: consensus seeking. When multiple agents work together, we are interested in how they can reach a consensus through inter-agent negotiation. To that end, this work studies a consensus-seeking task where the state of each agent is a numerical value and they negotiate with each other to reach a consensus value. It is revealed that when not explicitly directed on which strategy should be adopted, the LLM-driven agents primarily use the average strategy for consensus seeking although they may occasionally use some other strategies. Moreover, this work analyzes the impact of the agent number, agent personality, and network topology on the negotiation process. The findings reported in this work can potentially lay the foundations for understanding the behaviors of LLM-driven multi-
    

