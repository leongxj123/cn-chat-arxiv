# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards Human-Centered Construction Robotics: An RL-Driven Companion Robot For Contextually Assisting Carpentry Workers](https://arxiv.org/abs/2403.19060) | 本文提出了一种人类中心的建筑机器人方法，通过强化学习驱动的助手机器人为木工劳动者提供环境上下文协助，推进了机器人在建筑中的应用。 |
| [^2] | [A Scalable and Parallelizable Digital Twin Framework for Sustainable Sim2Real Transition of Multi-Agent Reinforcement Learning Systems](https://arxiv.org/abs/2403.10996) | 提出了一个可持续的多智能体深度强化学习框架，利用分散的学习架构，来解决交通路口穿越和自主赛车等问题 |
| [^3] | [Fully Spiking Neural Network for Legged Robots](https://arxiv.org/abs/2310.05022) | 本文将新型全脉冲神经网络（SNN）成功应用于处理腿式机器人，在各种模拟地形中取得了杰出结果。 |
| [^4] | [DRIFT: Deep Reinforcement Learning for Intelligent Floating Platforms Trajectories.](http://arxiv.org/abs/2310.04266) | 本研究提出了一种新的基于深度强化学习的套件，用于控制浮动平台的轨迹。通过训练精确操作策略以应对动态和不可预测的条件，解决了控制浮动平台中的不确定性问题。该套件具有稳健性、适应性和可传递性，并提供了丰富的可视化选项和与真实世界机器人系统集成的能力。 |
| [^5] | [DoReMi: Grounding Language Model by Detecting and Recovering from Plan-Execution Misalignment.](http://arxiv.org/abs/2307.00329) | DoReMi是一种新颖的语言模型基础架构，通过检测和修复计划与执行之间的不一致性来实现语言模型的基础。该架构利用视觉问答模型检查约束条件以发现不一致，并调用语言模型进行重新规划以实现恢复。 |
| [^6] | [Bringing robotics taxonomies to continuous domains via GPLVM on hyperbolic manifolds.](http://arxiv.org/abs/2210.01672) | 本论文提出了一种通过在双曲流形上使用GPLVM来在连续领域中应用机器人分类法的方法，通过捕捉相关层次结构的双曲嵌入来建模分类数据，并采用图形先验和保持距离的后向约束来实现分类法结构的纳入。 |

# 详细

[^1]: 人类中心施工机器人：基于强化学习的助手机器人为木工劳动者提供环境上下文协助

    Towards Human-Centered Construction Robotics: An RL-Driven Companion Robot For Contextually Assisting Carpentry Workers

    [https://arxiv.org/abs/2403.19060](https://arxiv.org/abs/2403.19060)

    本文提出了一种人类中心的建筑机器人方法，通过强化学习驱动的助手机器人为木工劳动者提供环境上下文协助，推进了机器人在建筑中的应用。

    

    在这个充满活力的建筑行业中，传统的机器人集成主要集中在自动化特定任务，通常忽略了建筑工作流程中人类因素的复杂性和变化性。本文提出了一种以人为本的方法，设计了一个“工作伴侣漫游器”，旨在协助建筑工人完成其现有实践，旨在增强安全性和工作流程的流畅性，同时尊重建筑劳动的技术性质。我们对在木工模板工程中部署机器人系统进行了深入研究，展示了一个原型，通过环境相关的强化学习（RL）驱动模块化框架，重点强调了在动态环境中的机动性、安全性和舒适的工人-机器人协作。我们的研究推进了机器人在建筑中的应用，倡导协作模型，其中自适应机器人支持而不是取代人类，强调了交互式的潜力。

    arXiv:2403.19060v1 Announce Type: cross  Abstract: In the dynamic construction industry, traditional robotic integration has primarily focused on automating specific tasks, often overlooking the complexity and variability of human aspects in construction workflows. This paper introduces a human-centered approach with a ``work companion rover" designed to assist construction workers within their existing practices, aiming to enhance safety and workflow fluency while respecting construction labor's skilled nature. We conduct an in-depth study on deploying a robotic system in carpentry formwork, showcasing a prototype that emphasizes mobility, safety, and comfortable worker-robot collaboration in dynamic environments through a contextual Reinforcement Learning (RL)-driven modular framework. Our research advances robotic applications in construction, advocating for collaborative models where adaptive robots support rather than replace humans, underscoring the potential for an interactive a
    
[^2]: 一个可扩展且可并行化的数字孪生框架，用于多智能体强化学习系统可持续Sim2Real转换

    A Scalable and Parallelizable Digital Twin Framework for Sustainable Sim2Real Transition of Multi-Agent Reinforcement Learning Systems

    [https://arxiv.org/abs/2403.10996](https://arxiv.org/abs/2403.10996)

    提出了一个可持续的多智能体深度强化学习框架，利用分散的学习架构，来解决交通路口穿越和自主赛车等问题

    

    本工作提出了一个可持续的多智能体深度强化学习框架，能够选择性地按需扩展并行化训练工作负载，并利用最少的硬件资源将训练好的策略从模拟环境转移到现实世界。我们引入了AutoDRIVE生态系统作为一个启动数字孪生框架，用于训练、部署和转移合作和竞争的多智能体强化学习策略从模拟环境到现实世界。具体来说，我们首先探究了4台合作车辆(Nigel)在单智能体和多智能体学习环境中共享有限状态信息的交叉遍历问题，采用了一种通用策略方法。然后，我们使用个体策略方法研究了2辆车(F1TENTH)的对抗性自主赛车问题。在任何一组实验中，我们采用了去中心化学习架构，这允许对策略进行有力的训练和测试。

    arXiv:2403.10996v1 Announce Type: cross  Abstract: This work presents a sustainable multi-agent deep reinforcement learning framework capable of selectively scaling parallelized training workloads on-demand, and transferring the trained policies from simulation to reality using minimal hardware resources. We introduce AutoDRIVE Ecosystem as an enabling digital twin framework to train, deploy, and transfer cooperative as well as competitive multi-agent reinforcement learning policies from simulation to reality. Particularly, we first investigate an intersection traversal problem of 4 cooperative vehicles (Nigel) that share limited state information in single as well as multi-agent learning settings using a common policy approach. We then investigate an adversarial autonomous racing problem of 2 vehicles (F1TENTH) using an individual policy approach. In either set of experiments, a decentralized learning architecture was adopted, which allowed robust training and testing of the policies 
    
[^3]: 用于腿式机器人的全脉冲神经网络

    Fully Spiking Neural Network for Legged Robots

    [https://arxiv.org/abs/2310.05022](https://arxiv.org/abs/2310.05022)

    本文将新型全脉冲神经网络（SNN）成功应用于处理腿式机器人，在各种模拟地形中取得了杰出结果。

    

    近年来，基于深度强化学习的腿式机器人取得了显著进展。四足机器人展示了在复杂环境中完成具有挑战性任务的能力，并已部署在现实场景中以协助人类。同时，两足和类人机器人在各种高难度任务中取得了突破。本研究成功将一种新型脉冲神经网络（SNN）应用于处理腿式机器人，在一系列模拟地形中取得了出色的结果。

    arXiv:2310.05022v2 Announce Type: replace-cross  Abstract: In recent years, legged robots based on deep reinforcement learning have made remarkable progress. Quadruped robots have demonstrated the ability to complete challenging tasks in complex environments and have been deployed in real-world scenarios to assist humans. Simultaneously, bipedal and humanoid robots have achieved breakthroughs in various demanding tasks. Current reinforcement learning methods can utilize diverse robot bodies and historical information to perform actions. However, prior research has not emphasized the speed and energy consumption of network inference, as well as the biological significance of the neural networks themselves. Most of the networks employed are traditional artificial neural networks that utilize multilayer perceptrons (MLP). In this paper, we successfully apply a novel Spiking Neural Network (SNN) to process legged robots, achieving outstanding results across a range of simulated terrains. S
    
[^4]: DRIFT: 智能浮动平台轨迹的深度强化学习

    DRIFT: Deep Reinforcement Learning for Intelligent Floating Platforms Trajectories. (arXiv:2310.04266v1 [cs.RO])

    [http://arxiv.org/abs/2310.04266](http://arxiv.org/abs/2310.04266)

    本研究提出了一种新的基于深度强化学习的套件，用于控制浮动平台的轨迹。通过训练精确操作策略以应对动态和不可预测的条件，解决了控制浮动平台中的不确定性问题。该套件具有稳健性、适应性和可传递性，并提供了丰富的可视化选项和与真实世界机器人系统集成的能力。

    

    本研究介绍了一种基于深度强化学习的新型套件，用于控制模拟和真实环境中的浮动平台。浮动平台可作为多功能的测试平台，在地球上模拟微重力环境。我们的方法通过训练能够在动态和不可预测的条件下进行精确操作的策略，解决了控制此类平台中的系统和环境不确定性问题。利用最先进的深度强化学习技术，我们的套件实现了稳健性、适应性和从模拟到现实的良好可传递性。我们的深度强化学习（DRL）框架提供了快速训练时间、大规模测试能力、丰富的可视化选项以及与真实世界机器人系统集成的ROS绑定。除了策略开发，我们的套件还为研究人员提供了一个全面的平台，提供开放访问，网址为https://github.com/elharirymatteo/RANS/tree/ICRA24。

    This investigation introduces a novel deep reinforcement learning-based suite to control floating platforms in both simulated and real-world environments. Floating platforms serve as versatile test-beds to emulate microgravity environments on Earth. Our approach addresses the system and environmental uncertainties in controlling such platforms by training policies capable of precise maneuvers amid dynamic and unpredictable conditions. Leveraging state-of-the-art deep reinforcement learning techniques, our suite achieves robustness, adaptability, and good transferability from simulation to reality. Our Deep Reinforcement Learning (DRL) framework provides advantages such as fast training times, large-scale testing capabilities, rich visualization options, and ROS bindings for integration with real-world robotic systems. Beyond policy development, our suite provides a comprehensive platform for researchers, offering open-access at https://github.com/elharirymatteo/RANS/tree/ICRA24.
    
[^5]: DoReMi: 通过检测和修复计划执行不一致来实现语言模型的基础

    DoReMi: Grounding Language Model by Detecting and Recovering from Plan-Execution Misalignment. (arXiv:2307.00329v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2307.00329](http://arxiv.org/abs/2307.00329)

    DoReMi是一种新颖的语言模型基础架构，通过检测和修复计划与执行之间的不一致性来实现语言模型的基础。该架构利用视觉问答模型检查约束条件以发现不一致，并调用语言模型进行重新规划以实现恢复。

    

    大型语言模型包含大量的语义知识，并具备出色的理解和推理能力。先前的研究已经探索了如何将语言模型与机器人任务相结合，以确保语言模型生成的序列在逻辑上正确且可执行。然而，由于环境扰动或控制器设计的不完善，底层执行可能会偏离高级计划。在本文中，我们提出了一种名为DoReMi的新型语言模型基础架构，该架构能够及时检测和修复计划与执行之间的不一致性。具体而言，我们利用LLM进行规划，并生成计划步骤的约束条件。这些约束条件可以指示计划与执行之间的不一致性，并且我们使用视觉问答（VQA）模型在低层技能执行过程中检查约束条件。如果发生特定的不一致，我们的方法将调用语言模型重新规划以从中恢复。

    Large language models encode a vast amount of semantic knowledge and possess remarkable understanding and reasoning capabilities. Previous research has explored how to ground language models in robotic tasks to ensure that the sequences generated by the language model are both logically correct and practically executable. However, low-level execution may deviate from the high-level plan due to environmental perturbations or imperfect controller design. In this paper, we propose DoReMi, a novel language model grounding framework that enables immediate Detection and Recovery from Misalignments between plan and execution. Specifically, LLMs are leveraged for both planning and generating constraints for planned steps. These constraints can indicate plan-execution misalignments and we use a vision question answering (VQA) model to check constraints during low-level skill execution. If certain misalignment occurs, our method will call the language model to re-plan in order to recover from mi
    
[^6]: 通过在双曲流形上使用GPLVM将机器人分类带入连续领域

    Bringing robotics taxonomies to continuous domains via GPLVM on hyperbolic manifolds. (arXiv:2210.01672v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2210.01672](http://arxiv.org/abs/2210.01672)

    本论文提出了一种通过在双曲流形上使用GPLVM来在连续领域中应用机器人分类法的方法，通过捕捉相关层次结构的双曲嵌入来建模分类数据，并采用图形先验和保持距离的后向约束来实现分类法结构的纳入。

    

    机器人分类被用作将人类的移动和与环境互动的方式进行高层次的分层抽象。它们已被证明对于分析抓取、操纵技能和全身支撑姿势非常有用。尽管我们在设计层次结构和基础类别方面做出了大量努力，但它们在应用领域的使用仍然有限。这可能是因为缺乏填补分类层级结构和与其类别相关联的高维异构数据之间差距的计算模型。为了解决这个问题，我们建议通过捕捉相关层次结构的双曲嵌入来建模分类数据。我们通过构建一个新颖的高斯过程双曲潜变量模型来实现这一点，该模型通过图形先验和保持距离的后向约束将分类法结构纳入潜在空间中。我们在三个不同的机器人分类法上验证了我们的模型。

    Robotic taxonomies serve as high-level hierarchical abstractions that classify how humans move and interact with their environment. They have proven useful to analyse grasps, manipulation skills, and whole-body support poses. Despite substantial efforts devoted to design their hierarchy and underlying categories, their use in application fields remains limited. This may be attributed to the lack of computational models that fill the gap between the discrete hierarchical structure of the taxonomy and the high-dimensional heterogeneous data associated to its categories. To overcome this problem, we propose to model taxonomy data via hyperbolic embeddings that capture the associated hierarchical structure. We achieve this by formulating a novel Gaussian process hyperbolic latent variable model that incorporates the taxonomy structure through graph-based priors on the latent space and distance-preserving back constraints. We validate our model on three different robotics taxonomies to lear
    

