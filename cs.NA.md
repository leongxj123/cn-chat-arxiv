# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Monotone Generative Modeling via a Gromov-Monge Embedding.](http://arxiv.org/abs/2311.01375) | 该论文提出了一种使用Gromov-Monge嵌入的深度生成模型，通过识别数据背后的底层结构，并将其映射到低维潜空间中，解决了生成对抗网络（GAN）中对初始条件敏感性和模式崩溃的问题。 |
| [^2] | [Solving Differential-Algebraic Equations in Power Systems Dynamics with Neural Networks and Spatial Decomposition.](http://arxiv.org/abs/2303.10256) | 本文提出了一种使用神经网络和空间分解来近似电力系统动力学微分代数方程的方法，旨在加速仿真，提高数值稳定性和精度。 |

# 详细

[^1]: 通过Gromov-Monge嵌入实现单调生成建模

    Monotone Generative Modeling via a Gromov-Monge Embedding. (arXiv:2311.01375v1 [cs.LG])

    [http://arxiv.org/abs/2311.01375](http://arxiv.org/abs/2311.01375)

    该论文提出了一种使用Gromov-Monge嵌入的深度生成模型，通过识别数据背后的底层结构，并将其映射到低维潜空间中，解决了生成对抗网络（GAN）中对初始条件敏感性和模式崩溃的问题。

    

    生成对抗网络（GAN）是创建新内容的强大工具，但面临着对初始条件的敏感性和模式崩溃等挑战。为了解决这些问题，我们提出了一种利用Gromov-Monge嵌入（GME）的深度生成模型。它帮助识别数据背后的底层测度的低维结构，然后将其映射到保持几何性质的低维潜空间中的一个测度，并将其最优地传输到参考测度。通过GME的保持底层几何性质和生成映射的$c$-周期性单调性来保证它们。后一特性是确保更好的参数初始化鲁棒性和模式崩溃的第一步。数值实验证明了我们的方法在生成高质量图像、避免模式崩溃和对不同起始条件具有鲁棒性方面的有效性。

    Generative Adversarial Networks (GANs) are powerful tools for creating new content, but they face challenges such as sensitivity to starting conditions and mode collapse. To address these issues, we propose a deep generative model that utilizes the Gromov-Monge embedding (GME). It helps identify the low-dimensional structure of the underlying measure of the data and then maps it, while preserving its geometry, into a measure in a low-dimensional latent space, which is then optimally transported to the reference measure. We guarantee the preservation of the underlying geometry by the GME and $c$-cyclical monotonicity of the generative map, where $c$ is an intrinsic embedding cost employed by the GME. The latter property is a first step in guaranteeing better robustness to initialization of parameters and mode collapse. Numerical experiments demonstrate the effectiveness of our approach in generating high-quality images, avoiding mode collapse, and exhibiting robustness to different star
    
[^2]: 利用神经网络和空间分解在电力系统动力学中求解微分代数方程

    Solving Differential-Algebraic Equations in Power Systems Dynamics with Neural Networks and Spatial Decomposition. (arXiv:2303.10256v1 [eess.SY])

    [http://arxiv.org/abs/2303.10256](http://arxiv.org/abs/2303.10256)

    本文提出了一种使用神经网络和空间分解来近似电力系统动力学微分代数方程的方法，旨在加速仿真，提高数值稳定性和精度。

    

    电力系统的动力学由一组微分代数方程描述。时间域仿真用于理解系统动态的演变。由于系统的刚度需要使用精细离散化的时间步长，因此这些仿真可能具有计算代价较高的特点。通过增加允许的时间步长，我们旨在加快这样的仿真。本文使用观察结果，即尽管各个组件使用代数和微分方程来描述，但它们的耦合仅涉及代数方程的观察结果，利用神经网络（NN）来近似组件状态演变，从而产生快速、准确和数值稳定的近似器，使得可以使用更大的时间步长。为了解释网络对组件以及组件对网络的影响，NN将耦合代数变量的时间演化作为其预测的输入。我们最初使用空间分解方法来估计NN，其中系统被分成空间区域，每个区域有单独的NN估计器。我们将基于NN的仿真与传统的数值积分方案进行比较，以展示我们的方法的有效性。

    The dynamics of the power system are described by a system of differential-algebraic equations. Time-domain simulations are used to understand the evolution of the system dynamics. These simulations can be computationally expensive due to the stiffness of the system which requires the use of finely discretized time-steps. By increasing the allowable time-step size, we aim to accelerate such simulations. In this paper, we use the observation that even though the individual components are described using both algebraic and differential equations, their coupling only involves algebraic equations. Following this observation, we use Neural Networks (NNs) to approximate the components' state evolution, leading to fast, accurate, and numerically stable approximators, which enable larger time-steps. To account for effects of the network on the components and vice-versa, the NNs take the temporal evolution of the coupling algebraic variables as an input for their prediction. We initially estima
    

