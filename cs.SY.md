# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Local Control Barrier Functions for Safety Control of Hybrid Systems.](http://arxiv.org/abs/2401.14907) | 该论文提出了一种学习启用的方法，能够构建本地控制屏障函数，以保证广泛类别的非线性混合动力系统的安全性。该方法是高效的，对参考控制器的干预最小，适用于大规模系统，并通过实证评估和比较案例展示了其功效和灵活性。 |
| [^2] | [Powerformer: A Section-adaptive Transformer for Power Flow Adjustment.](http://arxiv.org/abs/2401.02771) | Powerformer是一种适应不同传输区段的变压器架构，用于学习稳健电力系统状态表示。它通过开发专用的区段自适应注意机制，并引入图神经网络传播和多因素注意机制来提供更加稳健的状态表示。在三个不同的电力系统场景上进行了广泛评估。 |
| [^3] | [Moving-Horizon Estimators for Hyperbolic and Parabolic PDEs in 1-D.](http://arxiv.org/abs/2401.02516) | 本文介绍了一种用于双曲和抛物型PDE的移动时域估计器，通过PDE反步法将难以解决的观测器PDE转化为可以明确解决的目标观测器PDE，从而实现了在实时环境下消除数值解观测器PDE的需求。 |

# 详细

[^1]: 学习本地控制屏障函数以实现混合系统的安全控制

    Learning Local Control Barrier Functions for Safety Control of Hybrid Systems. (arXiv:2401.14907v1 [cs.RO])

    [http://arxiv.org/abs/2401.14907](http://arxiv.org/abs/2401.14907)

    该论文提出了一种学习启用的方法，能够构建本地控制屏障函数，以保证广泛类别的非线性混合动力系统的安全性。该方法是高效的，对参考控制器的干预最小，适用于大规模系统，并通过实证评估和比较案例展示了其功效和灵活性。

    

    混合动力系统在实际的机器人应用中普遍存在，常涉及连续状态和离散状态切换。安全性是混合机器人系统的首要关注点。现有的混合系统的安全关键控制方法要么计算效率低下，对系统性能有损，要么仅适用于小规模系统。为了解决这些问题，在本文中，我们提出了一种学习启用的方法，用于构建本地控制屏障函数（CBFs），以保证广泛类别的非线性混合动力系统的安全性。最终，我们得到了一个安全的基于神经网络的CBF切换控制器。我们的方法在计算上高效，对任何参考控制器的干预最小，并适用于大规模系统。通过两个机器人示例（包括高维自主赛车案例），我们对我们的框架进行了实证评估，并与其他基于CBF的方法和模型预测控制进行了比较，展示了其功效和灵活性。

    Hybrid dynamical systems are ubiquitous as practical robotic applications often involve both continuous states and discrete switchings. Safety is a primary concern for hybrid robotic systems. Existing safety-critical control approaches for hybrid systems are either computationally inefficient, detrimental to system performance, or limited to small-scale systems. To amend these drawbacks, in this paper, we propose a learningenabled approach to construct local Control Barrier Functions (CBFs) to guarantee the safety of a wide class of nonlinear hybrid dynamical systems. The end result is a safe neural CBFbased switching controller. Our approach is computationally efficient, minimally invasive to any reference controller, and applicable to large-scale systems. We empirically evaluate our framework and demonstrate its efficacy and flexibility through two robotic examples including a high-dimensional autonomous racing case, against other CBF-based approaches and model predictive control.
    
[^2]: Powerformer：适应不同传输区段的变压器架构用于电力流调整

    Powerformer: A Section-adaptive Transformer for Power Flow Adjustment. (arXiv:2401.02771v1 [cs.LG])

    [http://arxiv.org/abs/2401.02771](http://arxiv.org/abs/2401.02771)

    Powerformer是一种适应不同传输区段的变压器架构，用于学习稳健电力系统状态表示。它通过开发专用的区段自适应注意机制，并引入图神经网络传播和多因素注意机制来提供更加稳健的状态表示。在三个不同的电力系统场景上进行了广泛评估。

    

    本文提出了一种专为学习稳健电力系统状态表示而量身定制的变压器架构，旨在优化跨不同传输区段的电力调度以进行电力流调整。具体而言，我们的提出的方法名为Powerformer，开发了一种专用的区段自适应注意机制，与传统变压器中使用的自注意分离开来。该机制有效地将电力系统状态与传输区段信息整合在一起，有助于开发稳健的状态表示。此外，通过考虑电力系统的图拓扑和母线节点的电气属性，我们引入了两种定制策略来进一步增强表达能力：图神经网络传播和多因素注意机制。我们在三个电力系统场景（包括IEEE 118节点系统、中国实际300节点系统和一个大型系统）上进行了广泛的评估。

    In this paper, we present a novel transformer architecture tailored for learning robust power system state representations, which strives to optimize power dispatch for the power flow adjustment across different transmission sections. Specifically, our proposed approach, named Powerformer, develops a dedicated section-adaptive attention mechanism, separating itself from the self-attention used in conventional transformers. This mechanism effectively integrates power system states with transmission section information, which facilitates the development of robust state representations. Furthermore, by considering the graph topology of power system and the electrical attributes of bus nodes, we introduce two customized strategies to further enhance the expressiveness: graph neural network propagation and multi-factor attention mechanism. Extensive evaluations are conducted on three power system scenarios, including the IEEE 118-bus system, a realistic 300-bus system in China, and a large-
    
[^3]: 在一维中为双曲和抛物型PDE引入移动时域估计器

    Moving-Horizon Estimators for Hyperbolic and Parabolic PDEs in 1-D. (arXiv:2401.02516v1 [eess.SY])

    [http://arxiv.org/abs/2401.02516](http://arxiv.org/abs/2401.02516)

    本文介绍了一种用于双曲和抛物型PDE的移动时域估计器，通过PDE反步法将难以解决的观测器PDE转化为可以明确解决的目标观测器PDE，从而实现了在实时环境下消除数值解观测器PDE的需求。

    

    对于PDE的观测器本身也是PDE。因此，使用这样的观测器产生实时估计是计算负担很重的。对于有限维和ODE系统，移动时域估计器（MHE）是一种操作符，其输出是状态估计，而输入是时域起始处的初始状态估计以及移动时间域内的测量输出和输入信号。在本文中，我们引入了用于解决PDE的MHE，以消除实时数值解观测器PDE的需求。我们使用PDE反步法实现了这一点，对于某些特定类别的双曲和抛物型PDE，它能够明确地产生移动时域状态估计。具体来说，为了明确地产生状态估计，我们使用了一个难以解决的观测器PDE的反步变换，将其转化为一个可以明确解决的目标观测器PDE。我们提出的MHE并不是新的观测器设计，而只是明确的MHE实现，它能够在移动时域内产生状态估计。

    Observers for PDEs are themselves PDEs. Therefore, producing real time estimates with such observers is computationally burdensome. For both finite-dimensional and ODE systems, moving-horizon estimators (MHE) are operators whose output is the state estimate, while their inputs are the initial state estimate at the beginning of the horizon as well as the measured output and input signals over the moving time horizon. In this paper we introduce MHEs for PDEs which remove the need for a numerical solution of an observer PDE in real time. We accomplish this using the PDE backstepping method which, for certain classes of both hyperbolic and parabolic PDEs, produces moving-horizon state estimates explicitly. Precisely, to explicitly produce the state estimates, we employ a backstepping transformation of a hard-to-solve observer PDE into a target observer PDE, which is explicitly solvable. The MHEs we propose are not new observer designs but simply the explicit MHE realizations, over a moving
    

