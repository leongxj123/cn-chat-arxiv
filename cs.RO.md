# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unravelling Responsibility for AI.](http://arxiv.org/abs/2308.02608) | 本文旨在解构人工智能责任的概念，提出了一种包含四种责任意义的有效组合，以支持对人工智能责任的实践推理。 |
| [^2] | [Towards Safe Autonomous Driving Policies using a Neuro-Symbolic Deep Reinforcement Learning Approach.](http://arxiv.org/abs/2307.01316) | 本文介绍了一种名为DRL with Symbolic Logics (DRLSL)的新颖神经符号无模型深度强化学习方法，旨在实现在真实环境中安全学习自主驾驶策略。该方法结合了深度强化学习和符号逻辑驱动的推理，允许通过与物理环境的实时交互来学习自主驾驶策略并确保安全性。 |
| [^3] | [Self-Tuning PID Control via a Hybrid Actor-Critic-Based Neural Structure for Quadcopter Control.](http://arxiv.org/abs/2307.01312) | 本研究提出了一种基于混合Actor-Critic神经结构的自整定PID控制器，用于四旋翼飞行器的姿态和高度控制，通过强化学习的方法调整PID增益，提高了系统的稳健性和可靠性。 |

# 详细

[^1]: 解构人工智能责任

    Unravelling Responsibility for AI. (arXiv:2308.02608v1 [cs.AI])

    [http://arxiv.org/abs/2308.02608](http://arxiv.org/abs/2308.02608)

    本文旨在解构人工智能责任的概念，提出了一种包含四种责任意义的有效组合，以支持对人工智能责任的实践推理。

    

    为了在涉及人工智能系统的复杂情况下合理思考责任应该放在何处，我们首先需要一个足够清晰和详细的跨学科词汇来谈论责任。责任是一种三元关系，涉及到一个行为者、一个事件和一种责任方式。作为一种有意识的为了支持对人工智能责任进行实践推理的“解构”责任概念的努力，本文采取了“行为者A对事件O负责”的三部分表述，并确定了A、负责、O的子类别的有效组合。这些有效组合我们称之为“责任串”，分为四种责任意义：角色责任、因果责任、法律责任和道德责任。我们通过两个运行示例进行了说明，一个涉及医疗AI系统，另一个涉及AV与行人的致命碰撞。

    To reason about where responsibility does and should lie in complex situations involving AI-enabled systems, we first need a sufficiently clear and detailed cross-disciplinary vocabulary for talking about responsibility. Responsibility is a triadic relation involving an actor, an occurrence, and a way of being responsible. As part of a conscious effort towards 'unravelling' the concept of responsibility to support practical reasoning about responsibility for AI, this paper takes the three-part formulation, 'Actor A is responsible for Occurrence O' and identifies valid combinations of subcategories of A, is responsible for, and O. These valid combinations - which we term "responsibility strings" - are grouped into four senses of responsibility: role-responsibility; causal responsibility; legal liability-responsibility; and moral responsibility. They are illustrated with two running examples, one involving a healthcare AI-based system and another the fatal collision of an AV with a pedes
    
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
    

