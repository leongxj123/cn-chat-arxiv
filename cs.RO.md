# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Survey for Foundation Models in Autonomous Driving](https://rss.arxiv.org/abs/2402.01105) | 本综述论文回顾了40多篇研究论文，总结了基于基础模型的自动驾驶在规划、仿真和关键任务方面的重要贡献，强调了大型语言模型的推理和翻译能力，视觉基础模型在物体检测和驾驶场景创建方面的应用，以及多模态基础模型的视觉理解和空间推理能力。 |
| [^2] | [PhysORD: A Neuro-Symbolic Approach for Physics-infused Motion Prediction in Off-road Driving](https://arxiv.org/abs/2404.01596) | PhysORD是一种神经符号方法，将物理定律融入神经模型中，显著提高了在越野驾驶中的运动预测泛化能力。 |
| [^3] | [A PPO-based DRL Auto-Tuning Nonlinear PID Drone Controller for Robust Autonomous Flights](https://arxiv.org/abs/2404.00204) | 该项目将非线性深度强化学习（DRL）代理引入无人机控制中，取代传统线性PID控制器，实现了无缝过渡、提高响应速度和稳定性，同时结合PPO策略训练DRL代理，并利用高精度跟踪系统提高自主飞行精度。 |
| [^4] | [Bridging the Sim-to-Real Gap with Bayesian Inference](https://arxiv.org/abs/2403.16644) | 提出了SIM-FSVGD方法，通过利用低保真度的物理先验，成功缩小模拟到现实的差距，能够在低数据量的情况下学习准确的动力学，并在实验中展示了在高性能赛车系统上的有效性。 |
| [^5] | [Single-Agent Actor Critic for Decentralized Cooperative Driving](https://arxiv.org/abs/2403.11914) | 提出了一种新颖的单Agent Actor Critic模型，旨在利用单Agent强化学习学习自主车辆的去中心化合作驾驶策略，并通过对各种交通场景的广泛评估展现了改善道路系统内不同瓶颈位置交通流量的巨大潜力。 |
| [^6] | [Globally Stable Neural Imitation Policies](https://arxiv.org/abs/2403.04118) | 提出了稳定神经动力系统（SNDS）的仿真学习制度，可生成具有正式稳定性保证的政策，并通过联合训练政策和其对应的李亚普诺夫候选者确保全局稳定性。 |
| [^7] | [Distilling Knowledge for Short-to-Long Term Trajectory Prediction.](http://arxiv.org/abs/2305.08553) | 本文提出了一种新的方法Di-Long，用于解决长期轨迹预测中越来越不确定和不可预测的问题。该方法利用蒸馏短期轨迹模型预测器来指导训练过程中的长期轨迹预测学生网络。学生网络观察短序列并预测长轨迹，教师网络观察更长序列并预测剩余短目标轨迹。 |
| [^8] | [Active Learning of Discrete-Time Dynamics for Uncertainty-Aware Model Predictive Control.](http://arxiv.org/abs/2210.12583) | 本文提出了一种用于主动学习非线性机器人系统动力学的方法，结合了离线和在线学习，能够在实时中准确推断模型动力学，并设计了一种不确定性感知模型预测控制器。 |

# 详细

[^1]: 自动驾驶领域基础模型综述

    A Survey for Foundation Models in Autonomous Driving

    [https://rss.arxiv.org/abs/2402.01105](https://rss.arxiv.org/abs/2402.01105)

    本综述论文回顾了40多篇研究论文，总结了基于基础模型的自动驾驶在规划、仿真和关键任务方面的重要贡献，强调了大型语言模型的推理和翻译能力，视觉基础模型在物体检测和驾驶场景创建方面的应用，以及多模态基础模型的视觉理解和空间推理能力。

    

    基于基础模型的出现，自然语言处理和计算机视觉领域发生了革命，为自动驾驶应用铺平了道路。本综述论文对40多篇研究论文进行了全面的回顾，展示了基础模型在提升自动驾驶中的作用。大型语言模型在自动驾驶的规划和仿真中发挥着重要作用，特别是通过其在推理、代码生成和翻译方面的能力。与此同时，视觉基础模型在关键任务中得到越来越广泛的应用，例如三维物体检测和跟踪，以及为仿真和测试创建逼真的驾驶场景。多模态基础模型可以整合多样的输入，展现出卓越的视觉理解和空间推理能力，对于端到端自动驾驶至关重要。本综述不仅提供了一个结构化的分类，根据模态和自动驾驶领域中的功能对基础模型进行分类，还深入研究了方法。

    The advent of foundation models has revolutionized the fields of natural language processing and computer vision, paving the way for their application in autonomous driving (AD). This survey presents a comprehensive review of more than 40 research papers, demonstrating the role of foundation models in enhancing AD. Large language models contribute to planning and simulation in AD, particularly through their proficiency in reasoning, code generation and translation. In parallel, vision foundation models are increasingly adapted for critical tasks such as 3D object detection and tracking, as well as creating realistic driving scenarios for simulation and testing. Multi-modal foundation models, integrating diverse inputs, exhibit exceptional visual understanding and spatial reasoning, crucial for end-to-end AD. This survey not only provides a structured taxonomy, categorizing foundation models based on their modalities and functionalities within the AD domain but also delves into the meth
    
[^2]: PhysORD：一种神经符号方法用于越野驾驶中注入物理学的运动预测

    PhysORD: A Neuro-Symbolic Approach for Physics-infused Motion Prediction in Off-road Driving

    [https://arxiv.org/abs/2404.01596](https://arxiv.org/abs/2404.01596)

    PhysORD是一种神经符号方法，将物理定律融入神经模型中，显著提高了在越野驾驶中的运动预测泛化能力。

    

    运动预测对于自主越野驾驶至关重要，但与在道路上驾驶相比，它面临着更多挑战，主要是由于车辆与地形之间复杂的相互作用。传统的基于物理的方法在准确建模动态系统和外部干扰方面遇到困难。相反，基于数据驱动的神经网络需要大量数据集，并且难以明确捕捉基本的物理定律，这很容易导致泛化能力差。通过融合这两种方法的优势，神经符号方法提出了一个有前途的方向。这些方法将物理定律嵌入神经模型中，可能显著提高泛化能力。然而，以往的研究都没有在现实世界的越野驾驶环境中进行评估。为了弥合这一差距，我们提出 PhysORD，这是一种神经符号方法，集成了守恒定律，即欧拉-拉格朗日方程。

    arXiv:2404.01596v1 Announce Type: cross  Abstract: Motion prediction is critical for autonomous off-road driving, however, it presents significantly more challenges than on-road driving because of the complex interaction between the vehicle and the terrain. Traditional physics-based approaches encounter difficulties in accurately modeling dynamic systems and external disturbance. In contrast, data-driven neural networks require extensive datasets and struggle with explicitly capturing the fundamental physical laws, which can easily lead to poor generalization. By merging the advantages of both methods, neuro-symbolic approaches present a promising direction. These methods embed physical laws into neural models, potentially significantly improving generalization capabilities. However, no prior works were evaluated in real-world settings for off-road driving. To bridge this gap, we present PhysORD, a neural-symbolic approach integrating the conservation law, i.e., the Euler-Lagrange equa
    
[^3]: 基于PPO的DRL自调PID非线性无人机控制器用于稳健自主飞行

    A PPO-based DRL Auto-Tuning Nonlinear PID Drone Controller for Robust Autonomous Flights

    [https://arxiv.org/abs/2404.00204](https://arxiv.org/abs/2404.00204)

    该项目将非线性深度强化学习（DRL）代理引入无人机控制中，取代传统线性PID控制器，实现了无缝过渡、提高响应速度和稳定性，同时结合PPO策略训练DRL代理，并利用高精度跟踪系统提高自主飞行精度。

    

    该项目旨在通过将非线性深度强化学习（DRL）代理作为传统线性比例积分微分（PID）控制器的替代品，从而彻底改变无人机飞行控制。主要目标是在手动和自主模式之间实现无缝过渡，提高响应速度和稳定性。我们在Gazebo模拟器中利用近端策略优化（PPO）强化学习策略来训练DRL代理。添加20000美元的室内Vicon跟踪系统提供<1mm的定位精度，显着提高了自主飞行精度。为了在最短的无碰撞轨迹中导航无人机，我们还建立了一个三维A*路径规划器并成功地将其实施到实际飞行中。

    arXiv:2404.00204v1 Announce Type: cross  Abstract: This project aims to revolutionize drone flight control by implementing a nonlinear Deep Reinforcement Learning (DRL) agent as a replacement for traditional linear Proportional Integral Derivative (PID) controllers. The primary objective is to seamlessly transition drones between manual and autonomous modes, enhancing responsiveness and stability. We utilize the Proximal Policy Optimization (PPO) reinforcement learning strategy within the Gazebo simulator to train the DRL agent. Adding a $20,000 indoor Vicon tracking system offers <1mm positioning accuracy, which significantly improves autonomous flight precision. To navigate the drone in the shortest collision-free trajectory, we also build a 3 dimensional A* path planner and implement it into the real flight successfully.
    
[^4]: 用贝叶斯推断缩小模拟到现实的差距

    Bridging the Sim-to-Real Gap with Bayesian Inference

    [https://arxiv.org/abs/2403.16644](https://arxiv.org/abs/2403.16644)

    提出了SIM-FSVGD方法，通过利用低保真度的物理先验，成功缩小模拟到现实的差距，能够在低数据量的情况下学习准确的动力学，并在实验中展示了在高性能赛车系统上的有效性。

    

    我们提出了SIM-FSVGD来从数据中学习机器人动力学。与传统方法相比，SIM-FSVGD利用低保真度的物理先验，如模拟器的形式，来规范神经网络模型的训练。在低数据情况下已经学习准确的动力学，SIM-FSVGD在更多数据可用时也能够扩展和表现出色。我们通过实验证明，学习隐式物理先验导致准确的平均模型估计以及精确的不确定性量化。我们展示了SIM-FSVGD在高性能RC赛车系统上缩小模拟到现实差距的有效性。使用基于模型的RL，我们展示了一个高度动态的停车转向动作，使用的数据量仅为现有技术的一半。

    arXiv:2403.16644v1 Announce Type: cross  Abstract: We present SIM-FSVGD for learning robot dynamics from data. As opposed to traditional methods, SIM-FSVGD leverages low-fidelity physical priors, e.g., in the form of simulators, to regularize the training of neural network models. While learning accurate dynamics already in the low data regime, SIM-FSVGD scales and excels also when more data is available. We empirically show that learning with implicit physical priors results in accurate mean model estimation as well as precise uncertainty quantification. We demonstrate the effectiveness of SIM-FSVGD in bridging the sim-to-real gap on a high-performance RC racecar system. Using model-based RL, we demonstrate a highly dynamic parking maneuver with drifting, using less than half the data compared to the state of the art.
    
[^5]: 单Agent Actor Critic用于去中心化合作驾驶

    Single-Agent Actor Critic for Decentralized Cooperative Driving

    [https://arxiv.org/abs/2403.11914](https://arxiv.org/abs/2403.11914)

    提出了一种新颖的单Agent Actor Critic模型，旨在利用单Agent强化学习学习自主车辆的去中心化合作驾驶策略，并通过对各种交通场景的广泛评估展现了改善道路系统内不同瓶颈位置交通流量的巨大潜力。

    

    主动交通管理结合自主车辆（AVs）承诺未来拥有减少拥堵和增强交通流量。然而，为实际应用开发算法需要解决连续交通流量和部分可观察性带来的挑战。为了弥合这一差距，推动主动交通管理领域朝着更大程度的去中心化发展，我们介绍了一个新颖的不对称actor-critic模型，旨在利用单Agent强化学习学习自主车辆的去中心化合作驾驶策略。我们的方法采用具有掩码的注意力神经网络来处理实际交通流量的动态特性和部分可观察性。通过在各种交通场景中针对基线控制器的广泛评估，我们的模型显示出在道路系统内不同瓶颈位置改善交通流量的巨大潜力。

    arXiv:2403.11914v1 Announce Type: new  Abstract: Active traffic management incorporating autonomous vehicles (AVs) promises a future with diminished congestion and enhanced traffic flow. However, developing algorithms for real-world application requires addressing the challenges posed by continuous traffic flow and partial observability. To bridge this gap and advance the field of active traffic management towards greater decentralization, we introduce a novel asymmetric actor-critic model aimed at learning decentralized cooperative driving policies for autonomous vehicles using single-agent reinforcement learning. Our approach employs attention neural networks with masking to handle the dynamic nature of real-world traffic flow and partial observability. Through extensive evaluations against baseline controllers across various traffic scenarios, our model shows great potential for improving traffic flow at diverse bottleneck locations within the road system. Additionally, we explore t
    
[^6]: 全局稳定的神经仿真政策

    Globally Stable Neural Imitation Policies

    [https://arxiv.org/abs/2403.04118](https://arxiv.org/abs/2403.04118)

    提出了稳定神经动力系统（SNDS）的仿真学习制度，可生成具有正式稳定性保证的政策，并通过联合训练政策和其对应的李亚普诺夫候选者确保全局稳定性。

    

    仿真学习提供了一种有效的方法，可以缓解从头开始在解决空间中学习政策的资源密集和耗时的特性。尽管结果政策可以可靠地模仿专家演示，但在状态空间的未探索区域中常常缺乏可预测性，这给在面对扰动时带来了重大安全问题。为了解决这些挑战，我们引入了稳定神经动力系统（SNDS），一种生成具有正式稳定性保证的政策的仿真学习制度。我们使用神经政策架构，促进基于李亚普诺夫定理的稳定性表示，并联合训练政策及其相应的李亚普诺夫候选者，以确保全局稳定性。我们通过在仿真中进行大量实验来验证我们的方法，并成功将经过训练的政策部署到现实世界的机械手臂上。实验结果表明，我们的SNDS方法相比现有方法具有更好的全局稳定性和鲁棒性。

    arXiv:2403.04118v1 Announce Type: cross  Abstract: Imitation learning presents an effective approach to alleviate the resource-intensive and time-consuming nature of policy learning from scratch in the solution space. Even though the resulting policy can mimic expert demonstrations reliably, it often lacks predictability in unexplored regions of the state-space, giving rise to significant safety concerns in the face of perturbations. To address these challenges, we introduce the Stable Neural Dynamical System (SNDS), an imitation learning regime which produces a policy with formal stability guarantees. We deploy a neural policy architecture that facilitates the representation of stability based on Lyapunov theorem, and jointly train the policy and its corresponding Lyapunov candidate to ensure global stability. We validate our approach by conducting extensive experiments in simulation and successfully deploying the trained policies on a real-world manipulator arm. The experimental resu
    
[^7]: 将知识蒸馏用于短期到长期轨迹预测

    Distilling Knowledge for Short-to-Long Term Trajectory Prediction. (arXiv:2305.08553v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2305.08553](http://arxiv.org/abs/2305.08553)

    本文提出了一种新的方法Di-Long，用于解决长期轨迹预测中越来越不确定和不可预测的问题。该方法利用蒸馏短期轨迹模型预测器来指导训练过程中的长期轨迹预测学生网络。学生网络观察短序列并预测长轨迹，教师网络观察更长序列并预测剩余短目标轨迹。

    

    长期轨迹预测是计算机视觉、机器学习和机器人领域中一个重要且具有挑战性的问题。其中一个基本困难在于随着时间范围的增长，轨迹的演变变得越来越不确定和不可预测，从而增加了问题的复杂性。为了克服这个问题，在本文中，我们提出了Di-Long，一种新的方法，它利用蒸馏短期轨迹模型预测器来指导训练过程中的长期轨迹预测学生网络。给定一个包含学生网络允许的观测序列和补充目标序列的总序列长度，我们让学生和教师对同一个完整轨迹定义两个不同但相关的任务：学生观察一个短序列并预测一个长轨迹，而教师观察一个更长的序列并预测剩下的短目标轨迹。

    Long-term trajectory forecasting is an important and challenging problem in the fields of computer vision, machine learning, and robotics. One fundamental difficulty stands in the evolution of the trajectory that becomes more and more uncertain and unpredictable as the time horizon grows, subsequently increasing the complexity of the problem. To overcome this issue, in this paper, we propose Di-Long, a new method that employs the distillation of a short-term trajectory model forecaster that guides a student network for long-term trajectory prediction during the training process. Given a total sequence length that comprehends the allowed observation for the student network and the complementary target sequence, we let the student and the teacher solve two different related tasks defined over the same full trajectory: the student observes a short sequence and predicts a long trajectory, whereas the teacher observes a longer sequence and predicts the remaining short target trajectory. The
    
[^8]: 用于不确定性感知模型预测控制的离散时间动力学的主动学习

    Active Learning of Discrete-Time Dynamics for Uncertainty-Aware Model Predictive Control. (arXiv:2210.12583v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2210.12583](http://arxiv.org/abs/2210.12583)

    本文提出了一种用于主动学习非线性机器人系统动力学的方法，结合了离线和在线学习，能够在实时中准确推断模型动力学，并设计了一种不确定性感知模型预测控制器。

    

    模型驱动的控制需要对系统动力学进行准确建模，以便在复杂和动态环境中精确且安全地控制机器人。此外，在操作条件变化的情况下，模型应该不断调整以弥补动力学变化。本文提出了一种主动学习方法来主动建模非线性机器人系统的动力学。我们结合了离线学习以往经验和在线学习当前机器人与未知环境的交互。这两个因素使得学习过程高效且自适应，能够在实时中准确推断模型动力学，即使在大大不同于训练分布的操作范围内也可行。此外，我们设计了一种对学习到的动力学的aleatoric（数据）不确定性启发式条件的不确定性感知模型预测控制器。该控制器可以主动选择最优的控制动作。

    Model-based control requires an accurate model of the system dynamics for precisely and safely controlling the robot in complex and dynamic environments. Moreover, in the presence of variations in the operating conditions, the model should be continuously refined to compensate for dynamics changes. In this paper, we present a self-supervised learning approach that actively models the dynamics of nonlinear robotic systems. We combine offline learning from past experience and online learning from current robot interaction with the unknown environment. These two ingredients enable a highly sample-efficient and adaptive learning process, capable of accurately inferring model dynamics in real-time even in operating regimes that greatly differ from the training distribution. Moreover, we design an uncertainty-aware model predictive controller that is heuristically conditioned to the aleatoric (data) uncertainty of the learned dynamics. This controller actively chooses the optimal control act
    

