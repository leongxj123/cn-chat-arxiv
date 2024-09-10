# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Incentive-Compatible Vertiport Reservation in Advanced Air Mobility: An Auction-Based Approach](https://arxiv.org/abs/2403.18166) | 本论文提出了一种激励兼容且个体理性的vertiport预订机制，旨在协调多个运营商操作的电动垂直起降（eVTOL）飞行器在vertiports之间移动，最大化所有运营商的总估值同时最小化vertiports的拥堵。 |
| [^2] | [Multistep Inverse Is Not All You Need](https://arxiv.org/abs/2403.11940) | 本研究考虑了控制问题中的观测空间到简化控制相关变量空间的编码器学习，AC-State方法是一个多步反向方法。 |
| [^3] | [DEEP-IoT: Downlink-Enhanced Efficient-Power Internet of Things](https://arxiv.org/abs/2403.00321) | DEEP-IoT通过“更多监听，更少传输”的策略，挑战和转变了传统的物联网通信模型，大幅降低能耗并提高设备寿命。 |
| [^4] | [Actively Learning Reinforcement Learning: A Stochastic Optimal Control Approach.](http://arxiv.org/abs/2309.10831) | 本文提供了一个框架来解决强化学习中的模型不确定性和计算成本高的问题，通过使用强化学习解决随机动态规划方程，生成的控制器能够主动学习模型不确定性，并确保安全性和实时学习。 |

# 详细

[^1]: 具有激励兼容性的先进空中移动中的Vertiport预订：基于拍卖的方法

    Incentive-Compatible Vertiport Reservation in Advanced Air Mobility: An Auction-Based Approach

    [https://arxiv.org/abs/2403.18166](https://arxiv.org/abs/2403.18166)

    本论文提出了一种激励兼容且个体理性的vertiport预订机制，旨在协调多个运营商操作的电动垂直起降（eVTOL）飞行器在vertiports之间移动，最大化所有运营商的总估值同时最小化vertiports的拥堵。

    

    预计未来先进空中移动（AAM）的崛起将成为一个数十亿美元的产业。市场机制被认为是AAM运营的一个重要组成部分，其中包括具有私人估值的异构运营商。本文研究了设计一种机制来协调由多个具有与其机群关联的异构估值的运营商操作的电动垂直起降（eVTOL）飞行器在vertiports之间移动的问题，同时强制执行vertiports的到达、离开和停车约束。特别是，我们提出了一种激励兼容且个体理性的vertiport预订机制，该机制最大化了一个社会福利度量标准，从而包括最大化所有运营商的总估值同时最小化vertiports的拥堵。此外，我们提高了设计这一机制的计算可处理性

    arXiv:2403.18166v1 Announce Type: cross  Abstract: The rise of advanced air mobility (AAM) is expected to become a multibillion-dollar industry in the near future. Market-based mechanisms are touted to be an integral part of AAM operations, which comprise heterogeneous operators with private valuations. In this work, we study the problem of designing a mechanism to coordinate the movement of electric vertical take-off and landing (eVTOL) aircraft, operated by multiple operators each having heterogeneous valuations associated with their fleet, between vertiports, while enforcing the arrival, departure, and parking constraints at vertiports. Particularly, we propose an incentive-compatible and individually rational vertiport reservation mechanism that maximizes a social welfare metric, which encapsulates the objective of maximizing the overall valuations of all operators while minimizing the congestion at vertiports. Additionally, we improve the computational tractability of designing th
    
[^2]: 多步反向不是你所需要的

    Multistep Inverse Is Not All You Need

    [https://arxiv.org/abs/2403.11940](https://arxiv.org/abs/2403.11940)

    本研究考虑了控制问题中的观测空间到简化控制相关变量空间的编码器学习，AC-State方法是一个多步反向方法。

    

    在真实世界的控制设置中，观测空间通常是不必要的高维且受到时间相关噪声的影响。然而，系统的可控动态通常远比原始观测数据的动态简单。因此，学习一个编码器将观测空间映射到一个包含控制相关变量的简化空间是可取的。本文考虑了由Efroni等人（2022年）首次提出的Ex-BMDP模型，该模型将观测可以分解为依赖于动作的潜在状态和独立于动作的时间相关噪声，并形式化了能够解决控制问题的场景。Lamb等人（2022年）提出了“AC-State”方法，用于学习一个编码器，从这些问题中的观测中提取包含完整依赖于动作的潜在状态表示。AC-State是一个多步反向方法，它使用路径中第一个和最后一个状态的编码来预测

    arXiv:2403.11940v1 Announce Type: new  Abstract: In real-world control settings, the observation space is often unnecessarily high-dimensional and subject to time-correlated noise. However, the controllable dynamics of the system are often far simpler than the dynamics of the raw observations. It is therefore desirable to learn an encoder to map the observation space to a simpler space of control-relevant variables. In this work, we consider the Ex-BMDP model, first proposed by Efroni et al. (2022), which formalizes control problems where observations can be factorized into an action-dependent latent state which evolves deterministically, and action-independent time-correlated noise. Lamb et al. (2022) proposes the "AC-State" method for learning an encoder to extract a complete action-dependent latent state representation from the observations in such problems. AC-State is a multistep-inverse method, in that it uses the encoding of the the first and last state in a path to predict the 
    
[^3]: DEEP-IoT: 下行增强型高效能物联网

    DEEP-IoT: Downlink-Enhanced Efficient-Power Internet of Things

    [https://arxiv.org/abs/2403.00321](https://arxiv.org/abs/2403.00321)

    DEEP-IoT通过“更多监听，更少传输”的策略，挑战和转变了传统的物联网通信模型，大幅降低能耗并提高设备寿命。

    

    本文介绍了DEEP-IoT，这是一种具有革命意义的通信范例，旨在重新定义物联网设备之间的通信方式。通过开创性的“更多监听，更少传输”的策略，DEEP-IoT挑战和转变了传统的发送方（物联网设备）为中心的通信模型，将接收方（接入点）作为关键角色，从而降低能耗并延长设备寿命。我们不仅概念化了DEEP-IoT，还通过在窄带系统中集成深度学习增强的反馈信道编码来实现它。模拟结果显示，IoT单元的运行寿命显著提高，比使用Turbo和Polar编码的传统系统提高了最多52.71%。这一进展标志着一种变革。

    arXiv:2403.00321v1 Announce Type: cross  Abstract: At the heart of the Internet of Things (IoT) -- a domain witnessing explosive growth -- the imperative for energy efficiency and the extension of device lifespans has never been more pressing. This paper presents DEEP-IoT, a revolutionary communication paradigm poised to redefine how IoT devices communicate. Through a pioneering "listen more, transmit less" strategy, DEEP-IoT challenges and transforms the traditional transmitter (IoT devices)-centric communication model to one where the receiver (the access point) play a pivotal role, thereby cutting down energy use and boosting device longevity. We not only conceptualize DEEP-IoT but also actualize it by integrating deep learning-enhanced feedback channel codes within a narrow-band system. Simulation results show a significant enhancement in the operational lifespan of IoT cells -- surpassing traditional systems using Turbo and Polar codes by up to 52.71%. This leap signifies a paradi
    
[^4]: 活动学习强化学习：一种随机最优控制方法的应用

    Actively Learning Reinforcement Learning: A Stochastic Optimal Control Approach. (arXiv:2309.10831v1 [cs.LG])

    [http://arxiv.org/abs/2309.10831](http://arxiv.org/abs/2309.10831)

    本文提供了一个框架来解决强化学习中的模型不确定性和计算成本高的问题，通过使用强化学习解决随机动态规划方程，生成的控制器能够主动学习模型不确定性，并确保安全性和实时学习。

    

    本文提供了一个框架来应对两个问题：（i）强化学习在模型不确定性方面的脆弱性，因为受控实验室/仿真和实际条件之间的不匹配，以及（ii）随机最优控制的计算成本过高。我们通过使用强化学习来解决随机动态规划方程来解决这两个问题。由此产生的强化学习控制器对于几种类型的约束条件是安全的，并且它可以主动学习模型不确定性。与探索和利用不同，探测和安全性由控制器自身自动实现，实现了实时学习。一个仿真示例证明了所提方法的有效性。

    In this paper we provide framework to cope with two problems: (i) the fragility of reinforcement learning due to modeling uncertainties because of the mismatch between controlled laboratory/simulation and real-world conditions and (ii) the prohibitive computational cost of stochastic optimal control. We approach both problems by using reinforcement learning to solve the stochastic dynamic programming equation. The resulting reinforcement learning controller is safe with respect to several types of constraints constraints and it can actively learn about the modeling uncertainties. Unlike exploration and exploitation, probing and safety are employed automatically by the controller itself, resulting real-time learning. A simulation example demonstrates the efficacy of the proposed approach.
    

