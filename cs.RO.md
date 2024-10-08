# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Survey of Optimization-based Task and Motion Planning: From Classical To Learning Approaches](https://arxiv.org/abs/2404.02817) | 本综述全面审视了基于优化的任务与运动规划，重点讨论了如何通过混合优化方法解决高度复杂、接触丰富的机器人运动和操作问题。 |
| [^2] | [SCANet: Correcting LEGO Assembly Errors with Self-Correct Assembly Network](https://arxiv.org/abs/2403.18195) | 介绍了单步组装错误校正任务和LEGO错误校正组装数据集（LEGO-ECA），提出了用于这一任务的自校正组装网络（SCANet）。 |
| [^3] | [Partially Observable Task and Motion Planning with Uncertainty and Risk Awareness](https://arxiv.org/abs/2403.10454) | 提出了一种具有不确定性和风险意识的TAMP策略（TAMPURA），能够高效地解决具有初始状态和动作结果不确定性的长时程规划问题，包括需要信息收集和避免不良和不可逆结果的问题。 |
| [^4] | [Towards Embedding Dynamic Personas in Interactive Robots: Masquerading Animated Social Kinematics (MASK)](https://arxiv.org/abs/2403.10041) | 该研究提出了一种名为MASK的机器人系统，通过非言语互动与观众进行互动，并利用有限状态机结构调整机器人行为，实现多种不同角色的动态表达。 |
| [^5] | [SELFI: Autonomous Self-Improvement with Reinforcement Learning for Social Navigation](https://arxiv.org/abs/2403.00991) | SELFI提出了一种在线学习方法，通过将在线无模型强化学习与离线基于模型的学习相结合，实现了机器人行为的快速改进，并在避撞和社交遵从行为方面取得了显著进展。 |
| [^6] | [Large-Scale Actionless Video Pre-Training via Discrete Diffusion for Efficient Policy Learning](https://arxiv.org/abs/2402.14407) | 利用离散扩散结合生成式预训练和少量机器人视频微调，实现从人类视频到机器人策略学习的知识迁移。 |
| [^7] | [NOD-TAMP: Multi-Step Manipulation Planning with Neural Object Descriptors.](http://arxiv.org/abs/2311.01530) | NOD-TAMP是一个基于TAMP的框架，利用神经物体描述符来解决复杂操纵任务中的泛化问题，通过从少量人类演示中提取轨迹并进行调整，有效解决了长时程任务的挑战，并在模拟环境中优于现有方法。 |
| [^8] | [Context-Conditional Navigation with a Learning-Based Terrain- and Robot-Aware Dynamics Model.](http://arxiv.org/abs/2307.09206) | 本文提出了一种名为TRADYN的概率地形和机器人感知前向动力学模型，能够适应在自主导航环境中的地形和机器人的变化，通过在模拟的二维导航环境中的实验证明，该模型在长视程轨迹预测任务中表现出较低的预测误差。 |
| [^9] | [Latent-Conditioned Policy Gradient for Multi-Objective Deep Reinforcement Learning.](http://arxiv.org/abs/2303.08909) | 该论文提出了一种新的多目标深度强化学习算法，通过策略梯度训练单个神经网络，以在单次训练运行中近似获取整个帕累托集，而不依赖于目标的线性标量化。 |

# 详细

[^1]: 优化型任务与运动规划综述：从经典到学习方法

    A Survey of Optimization-based Task and Motion Planning: From Classical To Learning Approaches

    [https://arxiv.org/abs/2404.02817](https://arxiv.org/abs/2404.02817)

    本综述全面审视了基于优化的任务与运动规划，重点讨论了如何通过混合优化方法解决高度复杂、接触丰富的机器人运动和操作问题。

    

    任务与运动规划（TAMP）将高层任务规划和低层运动规划结合起来，使机器人能够有效地推理解决长时域、动态任务。基于优化的TAMP专注于通过目标函数定义目标条件的混合优化方法，并且能够处理开放式目标、机器人动态和机器人与环境之间的物理交互。因此，基于优化的TAMP特别适合解决高度复杂、接触丰富的运动和操作问题。本综述全面审视了基于优化的TAMP，涵盖了（i）规划领域表示，包括动作描述语言和时态逻辑，（ii）TAMP各组件的个别解决策略，包括人工智能规划和轨迹优化（TO），以及（iii）基于逻辑的任务规划与基于模型的TO之间的动态相互作用。

    arXiv:2404.02817v1 Announce Type: cross  Abstract: Task and Motion Planning (TAMP) integrates high-level task planning and low-level motion planning to equip robots with the autonomy to effectively reason over long-horizon, dynamic tasks. Optimization-based TAMP focuses on hybrid optimization approaches that define goal conditions via objective functions and are capable of handling open-ended goals, robotic dynamics, and physical interaction between the robot and the environment. Therefore, optimization-based TAMP is particularly suited to solve highly complex, contact-rich locomotion and manipulation problems. This survey provides a comprehensive review on optimization-based TAMP, covering (i) planning domain representations, including action description languages and temporal logic, (ii) individual solution strategies for components of TAMP, including AI planning and trajectory optimization (TO), and (iii) the dynamic interplay between logic-based task planning and model-based TO. A 
    
[^2]: 用自校正组装网络纠正LEGO组装错误

    SCANet: Correcting LEGO Assembly Errors with Self-Correct Assembly Network

    [https://arxiv.org/abs/2403.18195](https://arxiv.org/abs/2403.18195)

    介绍了单步组装错误校正任务和LEGO错误校正组装数据集（LEGO-ECA），提出了用于这一任务的自校正组装网络（SCANet）。

    

    在机器人学和3D视觉中，自主组装面临着重大挑战，尤其是确保组装正确性。主流方法如MEPNet目前专注于基于手动提供的图像进行组件组装。然而，这些方法在需要长期规划的任务中往往难以取得满意的结果。在同一时间，我们观察到整合自校正模块可以在一定程度上缓解这些问题。受此问题启发，我们引入了单步组装错误校正任务，其中涉及识别和纠正组件组装错误。为支持这一领域的研究，我们提出了LEGO错误校正组装数据集（LEGO-ECA），包括用于组装步骤和组装失败实例的手动图像。此外，我们提出了自校正组装网络（SCANet），这是一种新颖的方法来解决这一任务。SCANet将组装的部件视为查询，

    arXiv:2403.18195v1 Announce Type: cross  Abstract: Autonomous assembly in robotics and 3D vision presents significant challenges, particularly in ensuring assembly correctness. Presently, predominant methods such as MEPNet focus on assembling components based on manually provided images. However, these approaches often fall short in achieving satisfactory results for tasks requiring long-term planning. Concurrently, we observe that integrating a self-correction module can partially alleviate such issues. Motivated by this concern, we introduce the single-step assembly error correction task, which involves identifying and rectifying misassembled components. To support research in this area, we present the LEGO Error Correction Assembly Dataset (LEGO-ECA), comprising manual images for assembly steps and instances of assembly failures. Additionally, we propose the Self-Correct Assembly Network (SCANet), a novel method to address this task. SCANet treats assembled components as queries, de
    
[^3]: 具有不确定性和风险意识的部分可观测任务和运动规划

    Partially Observable Task and Motion Planning with Uncertainty and Risk Awareness

    [https://arxiv.org/abs/2403.10454](https://arxiv.org/abs/2403.10454)

    提出了一种具有不确定性和风险意识的TAMP策略（TAMPURA），能够高效地解决具有初始状态和动作结果不确定性的长时程规划问题，包括需要信息收集和避免不良和不可逆结果的问题。

    

    集成任务和运动规划（TAMP）已被证明是一种有价值的方法，用于解决通用的长时程机器人操纵和导航问题。然而，典型的TAMP问题公式化假设完全可观测和确定性动作效果。这些假设限制了规划者获取信息和做出具有风险意识的决策的能力。我们提出了一种具有不确定性和风险意识的TAMP策略（TAMPURA），能够高效地解决具有初始状态和动作结果不确定性的长时程规划问题，包括需要信息收集和避免不良和不可逆结果的问题。我们的规划者在抽象任务级别和连续控制器级别均在存在不确定性条件下进行推理。鉴于一组在基本动作空间中运行的闭环目标驱动控制器，并描述了它们的前提条件和潜在能力，

    arXiv:2403.10454v1 Announce Type: cross  Abstract: Integrated task and motion planning (TAMP) has proven to be a valuable approach to generalizable long-horizon robotic manipulation and navigation problems. However, the typical TAMP problem formulation assumes full observability and deterministic action effects. These assumptions limit the ability of the planner to gather information and make decisions that are risk-aware. We propose a strategy for TAMP with Uncertainty and Risk Awareness (TAMPURA) that is capable of efficiently solving long-horizon planning problems with initial-state and action outcome uncertainty, including problems that require information gathering and avoiding undesirable and irreversible outcomes. Our planner reasons under uncertainty at both the abstract task level and continuous controller level. Given a set of closed-loop goal-conditioned controllers operating in the primitive action space and a description of their preconditions and potential capabilities, w
    
[^4]: 在交互式机器人中嵌入动态角色: 伪装动画社交运动学（MASK）

    Towards Embedding Dynamic Personas in Interactive Robots: Masquerading Animated Social Kinematics (MASK)

    [https://arxiv.org/abs/2403.10041](https://arxiv.org/abs/2403.10041)

    该研究提出了一种名为MASK的机器人系统，通过非言语互动与观众进行互动，并利用有限状态机结构调整机器人行为，实现多种不同角色的动态表达。

    

    本文介绍了一种创新的交互式机器人系统的设计和开发，以增强观众参与度，使用类似角色的人物形象。基于以角色为驱动的对话代理系统，本研究将该代理应用扩展到了物理领域，利用机器人提供更具沉浸感和互动体验。提出的系统名为Masquerading Animated Social Kinematics (MASK)，利用类人机器人通过非言语互动与客人互动，包括面部表情和手势。一种基于有限状态机结构的行为生成系统有效地调整机器人行为以传达不同的人物角色。MASK框架集成了感知引擎、行为选择引擎和综合动作库，使其能够在行为设计中需要最少人工干预地实现实时、动态互动。在用户对象研究过程中，探讨了系统的效果，并展示了其潜力以激发未来研究的兴趣。

    arXiv:2403.10041v1 Announce Type: cross  Abstract: This paper presents the design and development of an innovative interactive robotic system to enhance audience engagement using character-like personas. Built upon the foundations of persona-driven dialog agents, this work extends the agent application to the physical realm, employing robots to provide a more immersive and interactive experience. The proposed system, named the Masquerading Animated Social Kinematics (MASK), leverages an anthropomorphic robot which interacts with guests using non-verbal interactions, including facial expressions and gestures. A behavior generation system based upon a finite-state machine structure effectively conditions robotic behavior to convey distinct personas. The MASK framework integrates a perception engine, a behavior selection engine, and a comprehensive action library to enable real-time, dynamic interactions with minimal human intervention in behavior design. Throughout the user subject studi
    
[^5]: SELFI: 利用强化学习实现自主自我改进以进行社交导航

    SELFI: Autonomous Self-Improvement with Reinforcement Learning for Social Navigation

    [https://arxiv.org/abs/2403.00991](https://arxiv.org/abs/2403.00991)

    SELFI提出了一种在线学习方法，通过将在线无模型强化学习与离线基于模型的学习相结合，实现了机器人行为的快速改进，并在避撞和社交遵从行为方面取得了显著进展。

    

    自主自我改进的机器人通过与环境互动和经验积累来实现将是机器人系统在现实世界中投入使用的关键。本文提出了一种在线学习方法SELFI，利用在线机器人经验来快速高效地微调预训练的控制策略。SELFI将在线无模型强化学习应用于离线基于模型的学习之上，以发挥这两种学习范式的优点。具体来说，SELFI通过将离线预训练的模型学习目标与在线无模型强化学习中学习到的Q值相结合，稳定了在线学习过程。我们在多个现实环境中评估了SELFI，并报告了在避撞方面的改善，以及通过人类用户研究测量的更具社交遵从行为。SELFI使我们能够快速学习有用的机器人行为，减少了预先干预的人员干预。

    arXiv:2403.00991v1 Announce Type: cross  Abstract: Autonomous self-improving robots that interact and improve with experience are key to the real-world deployment of robotic systems. In this paper, we propose an online learning method, SELFI, that leverages online robot experience to rapidly fine-tune pre-trained control policies efficiently. SELFI applies online model-free reinforcement learning on top of offline model-based learning to bring out the best parts of both learning paradigms. Specifically, SELFI stabilizes the online learning process by incorporating the same model-based learning objective from offline pre-training into the Q-values learned with online model-free reinforcement learning. We evaluate SELFI in multiple real-world environments and report improvements in terms of collision avoidance, as well as more socially compliant behavior, measured by a human user study. SELFI enables us to quickly learn useful robotic behaviors with less human interventions such as pre-e
    
[^6]: 通过离散扩散进行大规模无动作视频预训练，以实现高效策略学习

    Large-Scale Actionless Video Pre-Training via Discrete Diffusion for Efficient Policy Learning

    [https://arxiv.org/abs/2402.14407](https://arxiv.org/abs/2402.14407)

    利用离散扩散结合生成式预训练和少量机器人视频微调，实现从人类视频到机器人策略学习的知识迁移。

    

    学习一个能够完成多个任务的通用实体代理面临挑战，主要源自缺乏有标记动作的机器人数据集。相比之下，存在大量捕捉复杂任务和与物理世界互动的人类视频。本文介绍了一种新颖框架，利用统一的离散扩散将人类视频上的生成式预训练与少量有标记机器人视频上的策略微调结合起来。我们首先将人类和机器人视频压缩成统一的视频标记。在预训练阶段，我们使用一个带有蒙版替换扩散策略的离散扩散模型来预测潜空间中的未来视频标记。在微调阶段，我们 h

    arXiv:2402.14407v1 Announce Type: new  Abstract: Learning a generalist embodied agent capable of completing multiple tasks poses challenges, primarily stemming from the scarcity of action-labeled robotic datasets. In contrast, a vast amount of human videos exist, capturing intricate tasks and interactions with the physical world. Promising prospects arise for utilizing actionless human videos for pre-training and transferring the knowledge to facilitate robot policy learning through limited robot demonstrations. In this paper, we introduce a novel framework that leverages a unified discrete diffusion to combine generative pre-training on human videos and policy fine-tuning on a small number of action-labeled robot videos. We start by compressing both human and robot videos into unified video tokens. In the pre-training stage, we employ a discrete diffusion model with a mask-and-replace diffusion strategy to predict future video tokens in the latent space. In the fine-tuning stage, we h
    
[^7]: NOD-TAMP:多步骤操纵规划中的神经物体描述符

    NOD-TAMP: Multi-Step Manipulation Planning with Neural Object Descriptors. (arXiv:2311.01530v1 [cs.RO])

    [http://arxiv.org/abs/2311.01530](http://arxiv.org/abs/2311.01530)

    NOD-TAMP是一个基于TAMP的框架，利用神经物体描述符来解决复杂操纵任务中的泛化问题，通过从少量人类演示中提取轨迹并进行调整，有效解决了长时程任务的挑战，并在模拟环境中优于现有方法。

    

    在家居和工厂环境中开发复杂操纵任务的智能机器人仍然具有挑战性，因为长时程任务、接触丰富的操纵以及需要在各种物体形状和场景布局之间进行泛化。虽然任务和运动规划（TAMP）提供了一个有希望的解决方案，但是它的假设，如动力学模型，限制了它在新颖背景中的适应性。神经物体描述符（NODs）在物体和场景泛化方面显示出了潜力，但在处理更广泛任务方面存在局限性。我们提出的基于TAMP的框架NOD-TAMP从少数人类演示中提取短的操纵轨迹，使用NOD特征来调整这些轨迹，并组合它们来解决广泛的长时程任务。在模拟环境中验证后，NOD-TAMP有效应对各种挑战，优于现有方法，建立了一个强有力的操纵规划框架。

    Developing intelligent robots for complex manipulation tasks in household and factory settings remains challenging due to long-horizon tasks, contact-rich manipulation, and the need to generalize across a wide variety of object shapes and scene layouts. While Task and Motion Planning (TAMP) offers a promising solution, its assumptions such as kinodynamic models limit applicability in novel contexts. Neural object descriptors (NODs) have shown promise in object and scene generalization but face limitations in addressing broader tasks. Our proposed TAMP-based framework, NOD-TAMP, extracts short manipulation trajectories from a handful of human demonstrations, adapts these trajectories using NOD features, and composes them to solve broad long-horizon tasks. Validated in a simulation environment, NOD-TAMP effectively tackles varied challenges and outperforms existing methods, establishing a cohesive framework for manipulation planning. For videos and other supplemental material, see the pr
    
[^8]: 带有基于学习的地形和机器人感知动力模型的上下文条件导航

    Context-Conditional Navigation with a Learning-Based Terrain- and Robot-Aware Dynamics Model. (arXiv:2307.09206v1 [cs.RO])

    [http://arxiv.org/abs/2307.09206](http://arxiv.org/abs/2307.09206)

    本文提出了一种名为TRADYN的概率地形和机器人感知前向动力学模型，能够适应在自主导航环境中的地形和机器人的变化，通过在模拟的二维导航环境中的实验证明，该模型在长视程轨迹预测任务中表现出较低的预测误差。

    

    在自主导航环境中，多个参数可能会发生变化。地形特性如摩擦系数可能会根据机器人的位置而随时间变化。此外，机器人的动力学可能会因不同负载、系统质量变化、磨损等原因而发生变化，从而改变执行器增益或关节摩擦力。自主代理应该能够适应这些变化。在本文中，我们开发了一种新颖的概率地形和机器人感知前向动力学模型，称为TRADYN，它能够适应上述变化。它基于基于神经过程的元学习前向动力学模型的最新进展。我们在模拟的二维导航环境中评估了我们的方法，使用了一个类似自行车的机器人和具有空间变化摩擦系数的不同地形布局。在我们的实验中，与非自适应方法相比，所提出的模型在长视程轨迹预测任务中表现出较低的预测误差。

    In autonomous navigation settings, several quantities can be subject to variations. Terrain properties such as friction coefficients may vary over time depending on the location of the robot. Also, the dynamics of the robot may change due to, e.g., different payloads, changing the system's mass, or wear and tear, changing actuator gains or joint friction. An autonomous agent should thus be able to adapt to such variations. In this paper, we develop a novel probabilistic, terrain- and robot-aware forward dynamics model, termed TRADYN, which is able to adapt to the above-mentioned variations. It builds on recent advances in meta-learning forward dynamics models based on Neural Processes. We evaluate our method in a simulated 2D navigation setting with a unicycle-like robot and different terrain layouts with spatially varying friction coefficients. In our experiments, the proposed model exhibits lower prediction error for the task of long-horizon trajectory prediction, compared to non-ada
    
[^9]: 多目标深度强化学习中的潜在条件策略梯度

    Latent-Conditioned Policy Gradient for Multi-Objective Deep Reinforcement Learning. (arXiv:2303.08909v1 [cs.LG])

    [http://arxiv.org/abs/2303.08909](http://arxiv.org/abs/2303.08909)

    该论文提出了一种新的多目标深度强化学习算法，通过策略梯度训练单个神经网络，以在单次训练运行中近似获取整个帕累托集，而不依赖于目标的线性标量化。

    

    在现实世界中进行序列决策通常需要找到平衡相互矛盾的目标的良好平衡点。一般来说，存在大量的帕累托最优策略，它们体现了不同的目标权衡模式，并且使用深度神经网络全面获得它们具有技术挑战性。在本文中，我们提出了一种新的多目标强化学习（MORL）算法，通过策略梯度训练单个神经网络，以在单次训练运行中近似获取整个帕累托集，而不依赖于目标的线性标量化。该方法适用于连续和离散的行动空间，并且不需要修改策略网络的设计。在基准环境中的数字实验证明了我们的方法与标准MORL基线相比的实用性和有效性。

    Sequential decision making in the real world often requires finding a good balance of conflicting objectives. In general, there exist a plethora of Pareto-optimal policies that embody different patterns of compromises between objectives, and it is technically challenging to obtain them exhaustively using deep neural networks. In this work, we propose a novel multi-objective reinforcement learning (MORL) algorithm that trains a single neural network via policy gradient to approximately obtain the entire Pareto set in a single run of training, without relying on linear scalarization of objectives. The proposed method works in both continuous and discrete action spaces with no design change of the policy network. Numerical experiments in benchmark environments demonstrate the practicality and efficacy of our approach in comparison to standard MORL baselines.
    

