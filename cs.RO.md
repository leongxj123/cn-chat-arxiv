# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Actor-Critic Physics-informed Neural Lyapunov Control](https://arxiv.org/abs/2403.08448) | 提出了一种新方法，通过使用祖博夫的偏微分方程（PDE）来训练神经网络控制器，以及对应的李雅普诺夫证书，以最大化区域吸引力，并尊重激励约束。 |
| [^2] | [Grasp, See and Place: Efficient Unknown Object Rearrangement with Policy Structure Prior](https://arxiv.org/abs/2402.15402) | 该论文提出了一种具有策略结构先验的高效未知物体重新排列系统，通过内外环的学习，实现了抓取、观察和放置在感知噪声中的优化。 |
| [^3] | [Hierarchical State Space Models for Continuous Sequence-to-Sequence Modeling](https://arxiv.org/abs/2402.10211) | 分层状态空间模型（HiSS）是一种针对连续序列到序列建模的技术，它利用堆叠的结构化状态空间模型来进行预测。 |

# 详细

[^1]: 《基于演员评论者物理信息神经李雅普诺夫控制》

    Actor-Critic Physics-informed Neural Lyapunov Control

    [https://arxiv.org/abs/2403.08448](https://arxiv.org/abs/2403.08448)

    提出了一种新方法，通过使用祖博夫的偏微分方程（PDE）来训练神经网络控制器，以及对应的李雅普诺夫证书，以最大化区域吸引力，并尊重激励约束。

    

    设计具有可证保证的稳定化任务控制策略是非线性控制中的一个长期问题。关键的性能指标是产生区域吸引力的大小，这基本上充当了封闭环系统对不确定性的弹性“边界”。本文提出了一种新方法，用于训练一个稳定的神经网络控制器以及其对应的李雅普诺夫证书，旨在最大化产生的区域吸引力，同时尊重激励约束。我们方法的关键之处在于使用祖博夫的偏微分方程（PDE），该方程精确地表征了给定控制策略的真实区域吸引力。我们的框架遵循演员评论者模式，我们在改进控制策略（演员）和学习祖博夫函数（评论者）之间交替进行。最后，我们通过调用SMT求解器计算出最大的可证区域吸引力。

    arXiv:2403.08448v1 Announce Type: new  Abstract: Designing control policies for stabilization tasks with provable guarantees is a long-standing problem in nonlinear control. A crucial performance metric is the size of the resulting region of attraction, which essentially serves as a robustness "margin" of the closed-loop system against uncertainties. In this paper, we propose a new method to train a stabilizing neural network controller along with its corresponding Lyapunov certificate, aiming to maximize the resulting region of attraction while respecting the actuation constraints. Crucial to our approach is the use of Zubov's Partial Differential Equation (PDE), which precisely characterizes the true region of attraction of a given control policy. Our framework follows an actor-critic pattern where we alternate between improving the control policy (actor) and learning a Zubov function (critic). Finally, we compute the largest certifiable region of attraction by invoking an SMT solver
    
[^2]: 抓取、观察和放置：具有策略结构先验的高效未知物体重新排列

    Grasp, See and Place: Efficient Unknown Object Rearrangement with Policy Structure Prior

    [https://arxiv.org/abs/2402.15402](https://arxiv.org/abs/2402.15402)

    该论文提出了一种具有策略结构先验的高效未知物体重新排列系统，通过内外环的学习，实现了抓取、观察和放置在感知噪声中的优化。

    

    我们关注未知物体重新排列任务，即机器人应重新配置物体到由RGB-D图像指定的期望目标配置中。最近的研究通过整合基于学习的感知模块来探索未知物体重新排列系统。然而，它们对感知误差敏感，并且较少关注任务级性能。本文旨在开发一个有效的系统，用于在感知噪声中重新排列未知物体。我们在理论上揭示了噪声感知如何以分离的方式影响抓取和放置，并展示这样的分离结构不容易改善任务的最优性。我们提出了具有分离结构作为先验的GSP，一个双环系统。对于内环，我们学习主动观察策略以提高放置的感知。对于外环，我们学习一个抓取策略，意识到物体匹配和抓取能力。

    arXiv:2402.15402v1 Announce Type: cross  Abstract: We focus on the task of unknown object rearrangement, where a robot is supposed to re-configure the objects into a desired goal configuration specified by an RGB-D image. Recent works explore unknown object rearrangement systems by incorporating learning-based perception modules. However, they are sensitive to perception error, and pay less attention to task-level performance. In this paper, we aim to develop an effective system for unknown object rearrangement amidst perception noise. We theoretically reveal the noisy perception impacts grasp and place in a decoupled way, and show such a decoupled structure is non-trivial to improve task optimality. We propose GSP, a dual-loop system with the decoupled structure as prior. For the inner loop, we learn an active seeing policy for self-confident object matching to improve the perception of place. For the outer loop, we learn a grasp policy aware of object matching and grasp capability gu
    
[^3]: 针对连续序列到序列建模的分层状态空间模型

    Hierarchical State Space Models for Continuous Sequence-to-Sequence Modeling

    [https://arxiv.org/abs/2402.10211](https://arxiv.org/abs/2402.10211)

    分层状态空间模型（HiSS）是一种针对连续序列到序列建模的技术，它利用堆叠的结构化状态空间模型来进行预测。

    

    arXiv:2402.10211v1 公告类型：新的 摘要：从原始感知数据的序列推理是从医疗设备到机器人领域中普遍存在的问题。这些问题常常涉及使用长序列的原始传感器数据（例如磁力计，压阻器）来预测理想的物理量序列（例如力量，惯性测量）。虽然经典方法对于局部线性预测问题非常有效，但在使用实际传感器时往往表现不佳。这些传感器通常是非线性的，受到外界变量（例如振动）的影响，并且表现出数据相关漂移。对于许多问题来说，预测任务受到稀缺标记数据集的限制，因为获取地面真实标签需要昂贵的设备。在这项工作中，我们提出了分层状态空间模型（HiSS），这是一种概念上简单、全新的连续顺序预测技术。HiSS将结构化的状态空间模型堆叠在一起，以创建一个暂定的预测模型。

    arXiv:2402.10211v1 Announce Type: new  Abstract: Reasoning from sequences of raw sensory data is a ubiquitous problem across fields ranging from medical devices to robotics. These problems often involve using long sequences of raw sensor data (e.g. magnetometers, piezoresistors) to predict sequences of desirable physical quantities (e.g. force, inertial measurements). While classical approaches are powerful for locally-linear prediction problems, they often fall short when using real-world sensors. These sensors are typically non-linear, are affected by extraneous variables (e.g. vibration), and exhibit data-dependent drift. For many problems, the prediction task is exacerbated by small labeled datasets since obtaining ground-truth labels requires expensive equipment. In this work, we present Hierarchical State-Space Models (HiSS), a conceptually simple, new technique for continuous sequential prediction. HiSS stacks structured state-space models on top of each other to create a tempor
    

