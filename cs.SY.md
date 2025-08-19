# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Zero-Sum Linear Quadratic Games with Improved Sample Complexity.](http://arxiv.org/abs/2309.04272) | 这项研究提出了改进样本复杂性的零和线性二次博弈，并发现了自然策略梯度方法的隐式正则化属性。在无模型参数知识的情况下，他们还提出了第一个多项式样本复杂性算法来达到Nash均衡。 |
| [^2] | [Towards Safe Autonomous Driving Policies using a Neuro-Symbolic Deep Reinforcement Learning Approach.](http://arxiv.org/abs/2307.01316) | 本文介绍了一种名为DRL with Symbolic Logics (DRLSL)的新颖神经符号无模型深度强化学习方法，旨在实现在真实环境中安全学习自主驾驶策略。该方法结合了深度强化学习和符号逻辑驱动的推理，允许通过与物理环境的实时交互来学习自主驾驶策略并确保安全性。 |
| [^3] | [Self-Tuning PID Control via a Hybrid Actor-Critic-Based Neural Structure for Quadcopter Control.](http://arxiv.org/abs/2307.01312) | 本研究提出了一种基于混合Actor-Critic神经结构的自整定PID控制器，用于四旋翼飞行器的姿态和高度控制，通过强化学习的方法调整PID增益，提高了系统的稳健性和可靠性。 |

# 详细

[^1]: 学习改进样本复杂性的零和线性二次博弈

    Learning Zero-Sum Linear Quadratic Games with Improved Sample Complexity. (arXiv:2309.04272v1 [eess.SY])

    [http://arxiv.org/abs/2309.04272](http://arxiv.org/abs/2309.04272)

    这项研究提出了改进样本复杂性的零和线性二次博弈，并发现了自然策略梯度方法的隐式正则化属性。在无模型参数知识的情况下，他们还提出了第一个多项式样本复杂性算法来达到Nash均衡。

    

    零和线性二次（LQ）博弈在最优控制中是基础性的，可以用于（i）风险敏感或鲁棒控制的动态博弈形式，或者（ii）作为连续状态-控制空间中两个竞争智能体的多智能体强化学习的基准设置。与广泛研究的单智能体线性二次调节器问题不同，零和LQ博弈涉及解决一个具有缺乏强制性的目标函数的具有挑战性的非凸非凹最小-最大问题。最近，张等人发现了自然策略梯度方法的隐式正则化属性，这对于安全关键的控制系统非常重要，因为它在学习过程中保持了控制器的鲁棒性。此外，在没有模型参数知识的模型无关设置中，张等人提出了第一个多项式样本复杂性算法，以达到Nash均衡的ε-邻域，同时保持理想的隐式正则化属性。

    Zero-sum Linear Quadratic (LQ) games are fundamental in optimal control and can be used (i) as a dynamic game formulation for risk-sensitive or robust control, or (ii) as a benchmark setting for multi-agent reinforcement learning with two competing agents in continuous state-control spaces. In contrast to the well-studied single-agent linear quadratic regulator problem, zero-sum LQ games entail solving a challenging nonconvex-nonconcave min-max problem with an objective function that lacks coercivity. Recently, Zhang et al. discovered an implicit regularization property of natural policy gradient methods which is crucial for safety-critical control systems since it preserves the robustness of the controller during learning. Moreover, in the model-free setting where the knowledge of model parameters is not available, Zhang et al. proposed the first polynomial sample complexity algorithm to reach an $\epsilon$-neighborhood of the Nash equilibrium while maintaining the desirable implicit 
    
[^2]: 用神经符号深度强化学习方法实现安全自主驾驶策略的研究

    Towards Safe Autonomous Driving Policies using a Neuro-Symbolic Deep Reinforcement Learning Approach. (arXiv:2307.01316v1 [cs.RO])

    [http://arxiv.org/abs/2307.01316](http://arxiv.org/abs/2307.01316)

    本文介绍了一种名为DRL with Symbolic Logics (DRLSL)的新颖神经符号无模型深度强化学习方法，旨在实现在真实环境中安全学习自主驾驶策略。该方法结合了深度强化学习和符号逻辑驱动的推理，允许通过与物理环境的实时交互来学习自主驾驶策略并确保安全性。

    

    自主驾驶中的动态驾驶环境和多样化道路使用者的存在给决策造成了巨大的挑战。深度强化学习(DRL)已成为解决这一问题的一种流行方法。然而，由于安全问题的限制，现有的DRL解决方案的应用主要局限于模拟环境，阻碍了它们在现实世界中的部署。为了克服这一局限，本文引入了一种新颖的神经符号无模型深度强化学习方法，称为带有符号逻辑的DRL(DRLSL)，它将DRL(从经验中学习)和符号一阶逻辑知识驱动的推理相结合，以实现在实际环境下安全学习自主驾驶的实时交互。这种创新的方法提供了一种通过积极与物理环境互动来学习自主驾驶政策并确保安全性的方式。我们使用高维度数据实现了自主驾驶的DRLSL框架。

    The dynamic nature of driving environments and the presence of diverse road users pose significant challenges for decision-making in autonomous driving. Deep reinforcement learning (DRL) has emerged as a popular approach to tackle this problem. However, the application of existing DRL solutions is mainly confined to simulated environments due to safety concerns, impeding their deployment in real-world. To overcome this limitation, this paper introduces a novel neuro-symbolic model-free DRL approach, called DRL with Symbolic Logics (DRLSL) that combines the strengths of DRL (learning from experience) and symbolic first-order logics knowledge-driven reasoning) to enable safe learning in real-time interactions of autonomous driving within real environments. This innovative approach provides a means to learn autonomous driving policies by actively engaging with the physical environment while ensuring safety. We have implemented the DRLSL framework in autonomous driving using the highD data
    
[^3]: 通过基于混合Actor-Critic神经结构的自整定PID控制器，实现四旋翼飞行器控制

    Self-Tuning PID Control via a Hybrid Actor-Critic-Based Neural Structure for Quadcopter Control. (arXiv:2307.01312v1 [eess.SY])

    [http://arxiv.org/abs/2307.01312](http://arxiv.org/abs/2307.01312)

    本研究提出了一种基于混合Actor-Critic神经结构的自整定PID控制器，用于四旋翼飞行器的姿态和高度控制，通过强化学习的方法调整PID增益，提高了系统的稳健性和可靠性。

    

    比例积分微分（PID）控制器被广泛应用于工业和实验过程中，现有的离线方法可以用于调整PID增益。然而，由于模型参数的不确定性和外部干扰的存在，实际系统（如四旋翼飞行器）需要更稳健可靠的PID控制器。本研究探讨了一种使用强化学习的神经网络来实现四旋翼飞行器姿态和高度控制的自整定PID控制器。采用了增量式PID控制器，并仅对可变增益进行了调整。为了调整动态增益，使用了一种基于模型的无模型Actor-Critic混合神经结构，能够适当调整PID增益，同时充当最佳识别器。在调整和识别任务中，使用了一个具有两个隐藏层和Sigmoid激活函数的神经网络，并利用自适应动量（ADAM）优化器和反向传播算法进行学习。

    Proportional-Integrator-Derivative (PID) controller is used in a wide range of industrial and experimental processes. There are a couple of offline methods for tuning PID gains. However, due to the uncertainty of model parameters and external disturbances, real systems such as Quadrotors need more robust and reliable PID controllers. In this research, a self-tuning PID controller using a Reinforcement-Learning-based Neural Network for attitude and altitude control of a Quadrotor has been investigated. An Incremental PID, which contains static and dynamic gains, has been considered and only the variable gains have been tuned. To tune dynamic gains, a model-free actor-critic-based hybrid neural structure was used that was able to properly tune PID gains, and also has done the best as an identifier. In both tunning and identification tasks, a Neural Network with two hidden layers and sigmoid activation functions has been learned using Adaptive Momentum (ADAM) optimizer and Back-Propagatio
    

