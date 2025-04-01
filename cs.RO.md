# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Scalable Multi-modal Model Predictive Control via Duality-based Interaction Predictions](https://rss.arxiv.org/abs/2402.01116) | 我们提出了一个层级架构，通过使用对偶交互预测和精简的MPC问题，实现了可扩展的实时模型预测控制，在复杂的多模态交通场景中展示了12倍的速度提升。 |
| [^2] | [Verifiably Following Complex Robot Instructions with Foundation Models](https://arxiv.org/abs/2402.11498) | 提出了一种名为语言指令地面化运动规划（LIMP）系统，利用基础模型和时间逻辑生成指令条件的语义地图，使机器人能够可验证地遵循富有表现力和长期的指令，包括开放词汇参照和复杂的时空约束。 |

# 详细

[^1]: 可扩展多模型MPC的基于对偶交互预测的层级架构

    Scalable Multi-modal Model Predictive Control via Duality-based Interaction Predictions

    [https://rss.arxiv.org/abs/2402.01116](https://rss.arxiv.org/abs/2402.01116)

    我们提出了一个层级架构，通过使用对偶交互预测和精简的MPC问题，实现了可扩展的实时模型预测控制，在复杂的多模态交通场景中展示了12倍的速度提升。

    

    我们提出了一个层级架构，用于在复杂的多模态交通场景中实现可扩展的实时模型预测控制(MPC)。该架构由两个关键组件组成：1) RAID-Net，一种基于注意力机制的新颖循环神经网络，使用拉格朗日对偶性预测自动驾驶车辆与周围车辆之间在MPC预测范围内的相关交互；2) 一个简化的随机MPC问题，消除不相关的避碰约束，提高计算效率。我们的方法在一个模拟交通路口中演示，展示了解决运动规划问题的12倍速提升。您可以在这里找到展示该架构在多个复杂交通场景中的视频：https://youtu.be/-TcMeolCLWc

    We propose a hierarchical architecture designed for scalable real-time Model Predictive Control (MPC) in complex, multi-modal traffic scenarios. This architecture comprises two key components: 1) RAID-Net, a novel attention-based Recurrent Neural Network that predicts relevant interactions along the MPC prediction horizon between the autonomous vehicle and the surrounding vehicles using Lagrangian duality, and 2) a reduced Stochastic MPC problem that eliminates irrelevant collision avoidance constraints, enhancing computational efficiency. Our approach is demonstrated in a simulated traffic intersection with interactive surrounding vehicles, showcasing a 12x speed-up in solving the motion planning problem. A video demonstrating the proposed architecture in multiple complex traffic scenarios can be found here: https://youtu.be/-TcMeolCLWc
    
[^2]: 使用基础模型可验证地遵循复杂机器人指令

    Verifiably Following Complex Robot Instructions with Foundation Models

    [https://arxiv.org/abs/2402.11498](https://arxiv.org/abs/2402.11498)

    提出了一种名为语言指令地面化运动规划（LIMP）系统，利用基础模型和时间逻辑生成指令条件的语义地图，使机器人能够可验证地遵循富有表现力和长期的指令，包括开放词汇参照和复杂的时空约束。

    

    让机器人能够遵循复杂的自然语言指令是一个重要但具有挑战性的问题。人们希望在指导机器人时能够灵活表达约束，指向任意地标并验证行为。相反，机器人必须将人类指令消除歧义，将指令参照物联系到真实世界中。我们提出了一种名为语言指令地面化运动规划（LIMP）的系统，该系统利用基础模型和时间逻辑生成指令条件的语义地图，使机器人能够可验证地遵循富有表现力和长期的指令，涵盖了开放词汇参照和复杂的时空约束。与先前在机器人任务执行中使用基础模型的方法相比，LIMP构建了一个可解释的指令表示，揭示了机器人与指导者预期动机的一致性，并实现了机器人行为的综合。

    arXiv:2402.11498v1 Announce Type: cross  Abstract: Enabling robots to follow complex natural language instructions is an important yet challenging problem. People want to flexibly express constraints, refer to arbitrary landmarks and verify behavior when instructing robots. Conversely, robots must disambiguate human instructions into specifications and ground instruction referents in the real world. We propose Language Instruction grounding for Motion Planning (LIMP), a system that leverages foundation models and temporal logics to generate instruction-conditioned semantic maps that enable robots to verifiably follow expressive and long-horizon instructions with open vocabulary referents and complex spatiotemporal constraints. In contrast to prior methods for using foundation models in robot task execution, LIMP constructs an explainable instruction representation that reveals the robot's alignment with an instructor's intended motives and affords the synthesis of robot behaviors that 
    

