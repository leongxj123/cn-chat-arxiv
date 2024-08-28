# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Riemannian Flow Matching Policy for Robot Motion Learning](https://arxiv.org/abs/2403.10672) | RFMP是一种新颖的模型，利用流匹配的优势在机器人视觉运动策略中具有高效训练和推断能力，并通过融合黎曼流形上的几何意识，提供更平滑的动作轨迹。 |
| [^2] | [Structured Deep Neural Networks-Based Backstepping Trajectory Tracking Control for Lagrangian Systems](https://arxiv.org/abs/2403.00381) | 提出了一种基于结构化DNN的控制器，通过设计神经网络结构确保闭环稳定性，并进一步优化参数以实现改进的控制性能，同时提供了关于跟踪误差的明确上限。 |
| [^3] | [A Comprehensive Survey of Cross-Domain Policy Transfer for Embodied Agents](https://arxiv.org/abs/2402.04580) | 这篇论文综述了机器人跨领域策略转移方法，讨论了从目标领域采集无偏数据的挑战，以及从源领域获取数据的成本效益性。同时，总结了不同问题设置下的设计考虑和方法。 |

# 详细

[^1]: 用于机器人运动学习的黎曼流匹配策略

    Riemannian Flow Matching Policy for Robot Motion Learning

    [https://arxiv.org/abs/2403.10672](https://arxiv.org/abs/2403.10672)

    RFMP是一种新颖的模型，利用流匹配的优势在机器人视觉运动策略中具有高效训练和推断能力，并通过融合黎曼流形上的几何意识，提供更平滑的动作轨迹。

    

    我们引入了黎曼流匹配策略（RFMP），这是一种新颖的模型，用于学习和合成机器人视觉运动策略。RFMP利用了流匹配方法的高效训练和推断能力。通过设计，RFMP继承了流匹配的优势：能够编码高维度多模态分布，在机器人任务中常见，并且具有非常简单和快速的推断过程。我们展示了RFMP在基于状态和基于视觉的机器人运动策略中的适用性。值得注意的是，由于机器人状态存在于黎曼流形上，RFMP在本质上融合了几何意识，这对于现实机器人任务至关重要。为了评估RFMP，我们进行了两个概念验证实验，将其性能与扩散策略进行了比较。尽管这两种方法都成功地学习了所考虑的任务，但我们的结果表明RFMP提供了更平滑的动作轨迹，显著地提高了性能。

    arXiv:2403.10672v1 Announce Type: cross  Abstract: We introduce Riemannian Flow Matching Policies (RFMP), a novel model for learning and synthesizing robot visuomotor policies. RFMP leverages the efficient training and inference capabilities of flow matching methods. By design, RFMP inherits the strengths of flow matching: the ability to encode high-dimensional multimodal distributions, commonly encountered in robotic tasks, and a very simple and fast inference process. We demonstrate the applicability of RFMP to both state-based and vision-conditioned robot motion policies. Notably, as the robot state resides on a Riemannian manifold, RFMP inherently incorporates geometric awareness, which is crucial for realistic robotic tasks. To evaluate RFMP, we conduct two proof-of-concept experiments, comparing its performance against Diffusion Policies. Although both approaches successfully learn the considered tasks, our results show that RFMP provides smoother action trajectories with signifi
    
[^2]: 基于结构化深度神经网络的拉格朗日系统反步轨迹跟踪控制

    Structured Deep Neural Networks-Based Backstepping Trajectory Tracking Control for Lagrangian Systems

    [https://arxiv.org/abs/2403.00381](https://arxiv.org/abs/2403.00381)

    提出了一种基于结构化DNN的控制器，通过设计神经网络结构确保闭环稳定性，并进一步优化参数以实现改进的控制性能，同时提供了关于跟踪误差的明确上限。

    

    深度神经网络（DNN）越来越多地被用于学习控制器，因为其出色的逼近能力。然而，它们的黑盒特性对闭环稳定性保证和性能分析构成了重要挑战。在本文中，我们引入了一种基于结构化DNN的控制器，用于采用反推技术实现拉格朗日系统的轨迹跟踪控制。通过适当设计神经网络结构，所提出的控制器可以确保任何兼容的神经网络参数实现闭环稳定性。此外，通过进一步优化神经网络参数，可以实现更好的控制性能。此外，我们提供了关于跟踪误差的明确上限，这允许我们通过适当选择控制参数来实现所需的跟踪性能。此外，当系统模型未知时，我们提出了一种改进的拉格朗日神经网络。

    arXiv:2403.00381v1 Announce Type: cross  Abstract: Deep neural networks (DNN) are increasingly being used to learn controllers due to their excellent approximation capabilities. However, their black-box nature poses significant challenges to closed-loop stability guarantees and performance analysis. In this paper, we introduce a structured DNN-based controller for the trajectory tracking control of Lagrangian systems using backing techniques. By properly designing neural network structures, the proposed controller can ensure closed-loop stability for any compatible neural network parameters. In addition, improved control performance can be achieved by further optimizing neural network parameters. Besides, we provide explicit upper bounds on tracking errors in terms of controller parameters, which allows us to achieve the desired tracking performance by properly selecting the controller parameters. Furthermore, when system models are unknown, we propose an improved Lagrangian neural net
    
[^3]: 机器人跨领域策略转移综合调查

    A Comprehensive Survey of Cross-Domain Policy Transfer for Embodied Agents

    [https://arxiv.org/abs/2402.04580](https://arxiv.org/abs/2402.04580)

    这篇论文综述了机器人跨领域策略转移方法，讨论了从目标领域采集无偏数据的挑战，以及从源领域获取数据的成本效益性。同时，总结了不同问题设置下的设计考虑和方法。

    

    机器学习和具身人工智能领域的蓬勃发展引发了对大量数据的需求增加。然而，由于昂贵的数据收集过程和严格的安全要求，从目标领域收集足够的无偏数据仍然是一个挑战。因此，研究人员经常采用易于获取的源领域数据（例如模拟和实验室环境），以实现成本效益的数据获取和快速模型迭代。然而，这些源领域的环境和具身方式可能与目标领域的特征相差很大，强调了有效的跨领域策略转移方法的需求。本文对现有的跨领域策略转移方法进行了系统综述。通过对领域差距的精细分类，我们总结了每个问题设置的总体见解和设计考虑。我们还就使用的关键方法进行了高层次讨论

    The burgeoning fields of robot learning and embodied AI have triggered an increasing demand for large quantities of data. However, collecting sufficient unbiased data from the target domain remains a challenge due to costly data collection processes and stringent safety requirements. Consequently, researchers often resort to data from easily accessible source domains, such as simulation and laboratory environments, for cost-effective data acquisition and rapid model iteration. Nevertheless, the environments and embodiments of these source domains can be quite different from their target domain counterparts, underscoring the need for effective cross-domain policy transfer approaches. In this paper, we conduct a systematic review of existing cross-domain policy transfer methods. Through a nuanced categorization of domain gaps, we encapsulate the overarching insights and design considerations of each problem setting. We also provide a high-level discussion about the key methodologies used
    

