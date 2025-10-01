# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal transmission expansion minimally reduces decarbonization costs of U.S. electricity](https://arxiv.org/abs/2402.14189) | 传输扩展对高比例可再生能源系统的好处远远超过传统发电系统，全国最佳计划下的传输扩展仅能将100%清洁系统的成本降低4%。 |
| [^2] | [RobustNeuralNetworks.jl: a Package for Machine Learning and Data-Driven Control with Certified Robustness.](http://arxiv.org/abs/2306.12612) | RobustNeuralNetworks.jl是一个用Julia编写的机器学习和数据驱动控制包，它通过自然满足用户定义的鲁棒性约束条件，实现了神经网络模型的构建。 |

# 详细

[^1]: 最佳传输扩展最小化降低美国电力碳减排成本

    Optimal transmission expansion minimally reduces decarbonization costs of U.S. electricity

    [https://arxiv.org/abs/2402.14189](https://arxiv.org/abs/2402.14189)

    传输扩展对高比例可再生能源系统的好处远远超过传统发电系统，全国最佳计划下的传输扩展仅能将100%清洁系统的成本降低4%。

    

    太阳能和风能与化石燃料具有竞争力，但它们的间歇性特性带来了挑战。陆地、风力和太阳资源在时间和地理上存在显著差异，这表明远距离输电可能特别有益。我们使用详细的开源模型，联合分析了美国三个主要互联网的最佳传输扩展、存储、发电和小时运营。在高可再生能源系统中，传输扩展提供的好处远远超过主要采用传统发电的系统。然而，尽管一个最佳的全国计划需要将当前的地区间输电容量增加至三倍以上，与仅依靠当前输电的方案相比，传输仅能将100%清洁系统的成本降低4%。仅在现有互联网之间扩展容量即可实现大部分节约。对能源存储和发电的调整也能实现一定程度上的节约。

    arXiv:2402.14189v1 Announce Type: new  Abstract: Solar and wind power are cost-competitive with fossil fuels, yet their intermittent nature presents challenges. Significant temporal and geographic differences in land, wind, and solar resources suggest that long-distance transmission could be particularly beneficial. Using a detailed, open-source model, we analyze optimal transmission expansion jointly with storage, generation, and hourly operations across the three primary interconnects in the United States. Transmission expansion offers far more benefits in a high-renewable system than in a system with mostly conventional generation. Yet while an optimal nationwide plan would have more than triple current interregional transmission, transmission decreases the cost of a 100% clean system by only 4% compared to a plan that relies solely on current transmission. Expanding capacity only within existing interconnects can achieve most of these savings. Adjustments to energy storage and gene
    
[^2]: RobustNeuralNetworks.jl：带有认证鲁棒性的机器学习和数据驱动控制包。

    RobustNeuralNetworks.jl: a Package for Machine Learning and Data-Driven Control with Certified Robustness. (arXiv:2306.12612v1 [cs.LG])

    [http://arxiv.org/abs/2306.12612](http://arxiv.org/abs/2306.12612)

    RobustNeuralNetworks.jl是一个用Julia编写的机器学习和数据驱动控制包，它通过自然满足用户定义的鲁棒性约束条件，实现了神经网络模型的构建。

    

    神经网络通常对于微小的输入扰动非常敏感，导致出现意外或脆弱的行为。本文介绍了RobustNeuralNetworks.jl：一个Julia包，用于构建神经网络模型，该模型自然地满足一组用户定义的鲁棒性约束条件。该包基于最近提出的Recurrent Equilibrium Network（REN）和Lipschitz-Bounded Deep Network（LBDN）模型类，并旨在直接与Julia最广泛使用的机器学习包Flux.jl接口。我们讨论了模型参数化背后的理论，概述了该包，并提供了一个教程，演示了其在图像分类、强化学习和非线性状态观测器设计中的应用。

    Neural networks are typically sensitive to small input perturbations, leading to unexpected or brittle behaviour. We present RobustNeuralNetworks.jl: a Julia package for neural network models that are constructed to naturally satisfy a set of user-defined robustness constraints. The package is based on the recently proposed Recurrent Equilibrium Network (REN) and Lipschitz-Bounded Deep Network (LBDN) model classes, and is designed to interface directly with Julia's most widely-used machine learning package, Flux.jl. We discuss the theory behind our model parameterization, give an overview of the package, and provide a tutorial demonstrating its use in image classification, reinforcement learning, and nonlinear state-observer design.
    

