# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A practical existence theorem for reduced order models based on convolutional autoencoders](https://arxiv.org/abs/2402.00435) | 本论文提出了基于卷积自编码器的降阶模型的实用存在定理，解决了在处理复杂非线性问题方面传统方法的不足，并讨论了如何学习潜在特征的挑战。 |
| [^2] | [Improving physics-informed DeepONets with hard constraints.](http://arxiv.org/abs/2309.07899) | 本研究提出了一种改进的物理信息深度学习策略，消除了对初始条件的学习需求，并确保在多次应用时得到的函数是连续的。 |

# 详细

[^1]: 基于卷积自编码器的降阶模型的实用存在定理

    A practical existence theorem for reduced order models based on convolutional autoencoders

    [https://arxiv.org/abs/2402.00435](https://arxiv.org/abs/2402.00435)

    本论文提出了基于卷积自编码器的降阶模型的实用存在定理，解决了在处理复杂非线性问题方面传统方法的不足，并讨论了如何学习潜在特征的挑战。

    

    近年来，深度学习在偏微分方程和降阶建模领域越发受欢迎，提供了基于物理知识的神经网络、神经算子、深度算子网络和深度学习降阶模型等强大的数据驱动技术。在这种情况下，基于卷积神经网络的深度自编码器表现出极高的效果，在处理复杂的非线性问题时，优于传统的降阶方法。然而，尽管基于CNN的自编码器在实践中取得了成功，但目前只有少数理论结果支持这些架构，通常以万能逼近定理的形式陈述。尤其是，尽管现有文献为设计卷积自编码器提供了指导方针，但学习潜在特征的后续挑战几乎没有被探究。

    In recent years, deep learning has gained increasing popularity in the fields of Partial Differential Equations (PDEs) and Reduced Order Modeling (ROM), providing domain practitioners with new powerful data-driven techniques such as Physics-Informed Neural Networks (PINNs), Neural Operators, Deep Operator Networks (DeepONets) and Deep-Learning based ROMs (DL-ROMs). In this context, deep autoencoders based on Convolutional Neural Networks (CNNs) have proven extremely effective, outperforming established techniques, such as the reduced basis method, when dealing with complex nonlinear problems. However, despite the empirical success of CNN-based autoencoders, there are only a few theoretical results supporting these architectures, usually stated in the form of universal approximation theorems. In particular, although the existing literature provides users with guidelines for designing convolutional autoencoders, the subsequent challenge of learning the latent features has been barely inv
    
[^2]: 改进具有硬约束的物理信息DeepONets

    Improving physics-informed DeepONets with hard constraints. (arXiv:2309.07899v1 [cs.LG])

    [http://arxiv.org/abs/2309.07899](http://arxiv.org/abs/2309.07899)

    本研究提出了一种改进的物理信息深度学习策略，消除了对初始条件的学习需求，并确保在多次应用时得到的函数是连续的。

    

    当前的物理信息神经网络（标准或操作符）仍然依赖于准确地学习所解决系统的初始条件。相比之下，标准的数值方法在不需要学习这些条件的情况下演化这些初始条件。在这项研究中，我们提出改进当前的物理信息深度学习策略，使得不需要学习初始条件，并且将其准确地表示在预测的解中。此外，该方法保证当将DeepONet多次应用于时间步长解上时，得到的函数是连续的。

    Current physics-informed (standard or operator) neural networks still rely on accurately learning the initial conditions of the system they are solving. In contrast, standard numerical methods evolve such initial conditions without needing to learn these. In this study, we propose to improve current physics-informed deep learning strategies such that initial conditions do not need to be learned and are represented exactly in the predicted solution. Moreover, this method guarantees that when a DeepONet is applied multiple times to time step a solution, the resulting function is continuous.
    

