# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Visual Quadrupedal Loco-Manipulation from Demonstrations](https://arxiv.org/abs/2403.20328) | 通过将走行操纵过程分解为低层强化学习控制器和高层行为克隆规划器，使四足机器人能够执行真实世界的操纵任务。 |
| [^2] | [DASA: Delay-Adaptive Multi-Agent Stochastic Approximation](https://arxiv.org/abs/2403.17247) | DASA算法是第一个收敛速度仅依赖于混合时间和平均延迟的算法，同时在马尔科夫采样下实现N倍的收敛加速。 |
| [^3] | [Automated System-level Testing of Unmanned Aerial Systems](https://arxiv.org/abs/2403.15857) | 本文提出了一种利用模型测试和人工智能技术自动生成、执行和评估无人机系统级测试的新颖方法。 |

# 详细

[^1]: 从示范学习视觉四足走行操纵

    Learning Visual Quadrupedal Loco-Manipulation from Demonstrations

    [https://arxiv.org/abs/2403.20328](https://arxiv.org/abs/2403.20328)

    通过将走行操纵过程分解为低层强化学习控制器和高层行为克隆规划器，使四足机器人能够执行真实世界的操纵任务。

    

    四足机器人逐渐被整合进人类环境。尽管四足机器人的行走能力不断增强，但它们在现实场景中与物体的互动仍然有限。为了让四足机器人能够执行真实世界的操纵任务，我们将走行操纵过程分解为基于强化学习（RL）的低层控制器和基于行为克隆（BC）的高层规划器。通过参数化操纵轨迹，我们同步上层和下层的努力，从而充分利用RL和BC的优势。我们的方法通过模拟和现实世界实验得到验证。

    arXiv:2403.20328v1 Announce Type: cross  Abstract: Quadruped robots are progressively being integrated into human environments. Despite the growing locomotion capabilities of quadrupedal robots, their interaction with objects in realistic scenes is still limited. While additional robotic arms on quadrupedal robots enable manipulating objects, they are sometimes redundant given that a quadruped robot is essentially a mobile unit equipped with four limbs, each possessing 3 degrees of freedom (DoFs). Hence, we aim to empower a quadruped robot to execute real-world manipulation tasks using only its legs. We decompose the loco-manipulation process into a low-level reinforcement learning (RL)-based controller and a high-level Behavior Cloning (BC)-based planner. By parameterizing the manipulation trajectory, we synchronize the efforts of the upper and lower layers, thereby leveraging the advantages of both RL and BC. Our approach is validated through simulations and real-world experiments, d
    
[^2]: DASA: 延迟自适应多智能体随机逼近

    DASA: Delay-Adaptive Multi-Agent Stochastic Approximation

    [https://arxiv.org/abs/2403.17247](https://arxiv.org/abs/2403.17247)

    DASA算法是第一个收敛速度仅依赖于混合时间和平均延迟的算法，同时在马尔科夫采样下实现N倍的收敛加速。

    

    我们考虑一种设置，其中$N$个智能体旨在通过并行操作并与中央服务器通信来加速一个常见的随机逼近（SA）问题。我们假定上行传输到服务器的传输受到异步和潜在无界时变延迟的影响。为了减轻延迟和落后者的影响，同时又能获得分布式计算的好处，我们提出了一种名为DASA的延迟自适应多智能体随机逼近算法。我们对DASA进行了有限时间分析，假设智能体的随机观测过程是独立马尔科夫链。与现有结果相比，DASA是第一个其收敛速度仅取决于混合时间$tmix$和平均延迟$\tau_{avg}$，同时在马尔科夫采样下实现N倍的收敛加速的算法。我们的工作对于各种SA应用是相关的。

    arXiv:2403.17247v1 Announce Type: new  Abstract: We consider a setting in which $N$ agents aim to speedup a common Stochastic Approximation (SA) problem by acting in parallel and communicating with a central server. We assume that the up-link transmissions to the server are subject to asynchronous and potentially unbounded time-varying delays. To mitigate the effect of delays and stragglers while reaping the benefits of distributed computation, we propose \texttt{DASA}, a Delay-Adaptive algorithm for multi-agent Stochastic Approximation. We provide a finite-time analysis of \texttt{DASA} assuming that the agents' stochastic observation processes are independent Markov chains. Significantly advancing existing results, \texttt{DASA} is the first algorithm whose convergence rate depends only on the mixing time $\tmix$ and on the average delay $\tau_{avg}$ while jointly achieving an $N$-fold convergence speedup under Markovian sampling. Our work is relevant for various SA applications, inc
    
[^3]: 无人机系统级测试的自动化系统

    Automated System-level Testing of Unmanned Aerial Systems

    [https://arxiv.org/abs/2403.15857](https://arxiv.org/abs/2403.15857)

    本文提出了一种利用模型测试和人工智能技术自动生成、执行和评估无人机系统级测试的新颖方法。

    

    无人机系统依赖于各种安全关键和任务关键的航空电子系统。国际安全标准的主要要求之一是对航空电子软件系统进行严格的系统级测试。当前工业实践是手动创建测试方案，使用模拟器手动/自动执行这些方案，并手动评估结果。本文提出了一种新颖的方法来自动化无人机系统级测试。所提出的方法(AITester)利用基于模型的测试和人工智能(AI)技术，自动生成、执行和评估各种测试方案。

    arXiv:2403.15857v1 Announce Type: cross  Abstract: Unmanned aerial systems (UAS) rely on various avionics systems that are safety-critical and mission-critical. A major requirement of international safety standards is to perform rigorous system-level testing of avionics software systems. The current industrial practice is to manually create test scenarios, manually/automatically execute these scenarios using simulators, and manually evaluate outcomes. The test scenarios typically consist of setting certain flight or environment conditions and testing the system under test in these settings. The state-of-the-art approaches for this purpose also require manual test scenario development and evaluation. In this paper, we propose a novel approach to automate the system-level testing of the UAS. The proposed approach (AITester) utilizes model-based testing and artificial intelligence (AI) techniques to automatically generate, execute, and evaluate various test scenarios. The test scenarios a
    

