# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Solving Differential-Algebraic Equations in Power Systems Dynamics with Neural Networks and Spatial Decomposition.](http://arxiv.org/abs/2303.10256) | 本文提出了一种使用神经网络和空间分解来近似电力系统动力学微分代数方程的方法，旨在加速仿真，提高数值稳定性和精度。 |
| [^2] | [Robust Pivoting Manipulation using Contact Implicit Bilevel Optimization.](http://arxiv.org/abs/2303.08965) | 本文使用接触隐式双层优化来规划支点操纵并增加鲁棒性，通过利用摩擦力来弥补物体和环境物理属性估计中的不准确性，以应对不确定性影响。 |

# 详细

[^1]: 利用神经网络和空间分解在电力系统动力学中求解微分代数方程

    Solving Differential-Algebraic Equations in Power Systems Dynamics with Neural Networks and Spatial Decomposition. (arXiv:2303.10256v1 [eess.SY])

    [http://arxiv.org/abs/2303.10256](http://arxiv.org/abs/2303.10256)

    本文提出了一种使用神经网络和空间分解来近似电力系统动力学微分代数方程的方法，旨在加速仿真，提高数值稳定性和精度。

    

    电力系统的动力学由一组微分代数方程描述。时间域仿真用于理解系统动态的演变。由于系统的刚度需要使用精细离散化的时间步长，因此这些仿真可能具有计算代价较高的特点。通过增加允许的时间步长，我们旨在加快这样的仿真。本文使用观察结果，即尽管各个组件使用代数和微分方程来描述，但它们的耦合仅涉及代数方程的观察结果，利用神经网络（NN）来近似组件状态演变，从而产生快速、准确和数值稳定的近似器，使得可以使用更大的时间步长。为了解释网络对组件以及组件对网络的影响，NN将耦合代数变量的时间演化作为其预测的输入。我们最初使用空间分解方法来估计NN，其中系统被分成空间区域，每个区域有单独的NN估计器。我们将基于NN的仿真与传统的数值积分方案进行比较，以展示我们的方法的有效性。

    The dynamics of the power system are described by a system of differential-algebraic equations. Time-domain simulations are used to understand the evolution of the system dynamics. These simulations can be computationally expensive due to the stiffness of the system which requires the use of finely discretized time-steps. By increasing the allowable time-step size, we aim to accelerate such simulations. In this paper, we use the observation that even though the individual components are described using both algebraic and differential equations, their coupling only involves algebraic equations. Following this observation, we use Neural Networks (NNs) to approximate the components' state evolution, leading to fast, accurate, and numerically stable approximators, which enable larger time-steps. To account for effects of the network on the components and vice-versa, the NNs take the temporal evolution of the coupling algebraic variables as an input for their prediction. We initially estima
    
[^2]: 使用接触隐式双层优化实现鲁棒的支点操作

    Robust Pivoting Manipulation using Contact Implicit Bilevel Optimization. (arXiv:2303.08965v1 [cs.RO])

    [http://arxiv.org/abs/2303.08965](http://arxiv.org/abs/2303.08965)

    本文使用接触隐式双层优化来规划支点操纵并增加鲁棒性，通过利用摩擦力来弥补物体和环境物理属性估计中的不准确性，以应对不确定性影响。

    

    通用操纵要求机器人能够与新物体和环境进行交互。这个要求使得操纵变得异常具有挑战性，因为机器人必须考虑到不确定因素下的复杂摩擦相互作用及物体和环境的物理属性估计的不准确性。本文研究了支点操作规划的鲁棒优化问题，提供了如何利用摩擦力来弥补物理特性估计中的不准确性的见解。在某些假设下，导出了摩擦力提供的支点操作稳定裕度的解析表达式。然后，在接触隐式双层优化(CIBO)框架中使用该裕度来优化轨迹 ，以增强对物体多个物理参数不确定性的鲁棒性。我们在实际机器人上的实验中，对于严重干扰的参数，分析了稳定裕度，并显示了优化轨迹的改善鲁棒性。

    Generalizable manipulation requires that robots be able to interact with novel objects and environment. This requirement makes manipulation extremely challenging as a robot has to reason about complex frictional interactions with uncertainty in physical properties of the object and the environment. In this paper, we study robust optimization for planning of pivoting manipulation in the presence of uncertainties. We present insights about how friction can be exploited to compensate for inaccuracies in the estimates of the physical properties during manipulation. Under certain assumptions, we derive analytical expressions for stability margin provided by friction during pivoting manipulation. This margin is then used in a Contact Implicit Bilevel Optimization (CIBO) framework to optimize a trajectory that maximizes this stability margin to provide robustness against uncertainty in several physical parameters of the object. We present analysis of the stability margin with respect to sever
    

