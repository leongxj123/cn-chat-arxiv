# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Unified Framework to Enforce, Discover, and Promote Symmetry in Machine Learning.](http://arxiv.org/abs/2311.00212) | 本文提供了一个统一的框架，通过三种方式将对称性融入机器学习模型：1. 强制已知的对称性；2. 发现未知的对称性；3. 在训练过程中促进对称性。 |
| [^2] | [A Data-facilitated Numerical Method for Richards Equation to Model Water Flow Dynamics in Soil.](http://arxiv.org/abs/2310.02806) | 本文提出了一种基于数据的Richards方程数值解法，称为D-GRW方法，通过整合自适应线性化方案、神经网络和全局随机游走，在有限体积离散化框架下，能够产生具有收敛性保证的Richards方程数值解，并在精度和质量守恒性能方面表现卓越。 |
| [^3] | [Auto-weighted Bayesian Physics-Informed Neural Networks and robust estimations for multitask inverse problems in pore-scale imaging of dissolution.](http://arxiv.org/abs/2308.12864) | 本文介绍了一种新的数据同化策略，可可靠地处理包含不确定度量化的孔隙尺度反应反问题。该方法结合了数据驱动和物理建模，确保了孔隙尺度模型的可靠校准。 |

# 详细

[^1]: 在机器学习中强制、发现和推动对称性的统一框架

    A Unified Framework to Enforce, Discover, and Promote Symmetry in Machine Learning. (arXiv:2311.00212v1 [cs.LG])

    [http://arxiv.org/abs/2311.00212](http://arxiv.org/abs/2311.00212)

    本文提供了一个统一的框架，通过三种方式将对称性融入机器学习模型：1. 强制已知的对称性；2. 发现未知的对称性；3. 在训练过程中促进对称性。

    

    对称性存在于自然界中，并在物理学和机器学习中扮演着越来越核心的角色。基本对称性，如庞加莱不变性，使在地球上实验室中发现的物理定律能够推广到宇宙的最远处。对称性对于在机器学习应用中实现这种推广能力至关重要。例如，在图像分类中的平移不变性允许使用参数更少的模型（如卷积神经网络）在较小的数据集上进行训练，并达到最先进的性能。本文提供了一个统一的理论和方法框架，用于在机器学习模型中以三种方式融入对称性：1. 在训练模型时强制已知的对称性；2. 发现给定模型或数据集的未知对称性；3. 在训练过程中通过学习打破用户指定的候选群体内的对称性来促进对称性。

    Symmetry is present throughout nature and continues to play an increasingly central role in physics and machine learning. Fundamental symmetries, such as Poincar\'{e} invariance, allow physical laws discovered in laboratories on Earth to be extrapolated to the farthest reaches of the universe. Symmetry is essential to achieving this extrapolatory power in machine learning applications. For example, translation invariance in image classification allows models with fewer parameters, such as convolutional neural networks, to be trained on smaller data sets and achieve state-of-the-art performance. In this paper, we provide a unifying theoretical and methodological framework for incorporating symmetry into machine learning models in three ways: 1. enforcing known symmetry when training a model; 2. discovering unknown symmetries of a given model or data set; and 3. promoting symmetry during training by learning a model that breaks symmetries within a user-specified group of candidates when 
    
[^2]: 一种基于数据的Richards方程数值解法，用于模拟土壤中的水流动力学

    A Data-facilitated Numerical Method for Richards Equation to Model Water Flow Dynamics in Soil. (arXiv:2310.02806v1 [math.NA])

    [http://arxiv.org/abs/2310.02806](http://arxiv.org/abs/2310.02806)

    本文提出了一种基于数据的Richards方程数值解法，称为D-GRW方法，通过整合自适应线性化方案、神经网络和全局随机游走，在有限体积离散化框架下，能够产生具有收敛性保证的Richards方程数值解，并在精度和质量守恒性能方面表现卓越。

    

    根区土壤湿度的监测对于精密农业、智能灌溉和干旱预防至关重要。通常通过求解Richards方程这样的水文模型来模拟土壤的时空水流动力学。在本文中，我们提出了一种新型的基于数据的Richards方程数值解法。这种数值解法被称为D-GRW（Data-facilitated global Random Walk）方法，它在有限体积离散化框架中协同地整合了自适应线性化方案、神经网络和全局随机游走，可以在合理的假设下产生精确的Richards方程数值解，并且具有收敛性保证。通过三个示例，我们展示和讨论了我们的D-GRW方法在精度和质量守恒性能方面的卓越表现，并将其与基准数值解法和商用软件进行了比较。

    Root-zone soil moisture monitoring is essential for precision agriculture, smart irrigation, and drought prevention. Modeling the spatiotemporal water flow dynamics in soil is typically achieved by solving a hydrological model, such as the Richards equation which is a highly nonlinear partial differential equation (PDE). In this paper, we present a novel data-facilitated numerical method for solving the mixed-form Richards equation. This numerical method, which we call the D-GRW (Data-facilitated global Random Walk) method, synergistically integrates adaptive linearization scheme, neural networks, and global random walk in a finite volume discretization framework to produce accurate numerical solutions of the Richards equation with guaranteed convergence under reasonable assumptions. Through three illustrative examples, we demonstrate and discuss the superior accuracy and mass conservation performance of our D-GRW method and compare it with benchmark numerical methods and commercial so
    
[^3]: 自动加权的贝叶斯物理信息神经网络和鲁棒估计在孔隙尺度溶解图像的多任务反问题中的应用

    Auto-weighted Bayesian Physics-Informed Neural Networks and robust estimations for multitask inverse problems in pore-scale imaging of dissolution. (arXiv:2308.12864v1 [cs.LG])

    [http://arxiv.org/abs/2308.12864](http://arxiv.org/abs/2308.12864)

    本文介绍了一种新的数据同化策略，可可靠地处理包含不确定度量化的孔隙尺度反应反问题。该方法结合了数据驱动和物理建模，确保了孔隙尺度模型的可靠校准。

    

    在这篇文章中，我们提出了一种新的数据同化策略，并展示了这种方法可以使我们能够可靠地处理包含不确定度量化的反应反问题。孔隙尺度的反应流动建模为研究宏观性质在动态过程中的演变提供了宝贵的机会。然而，它们受到相关的X射线微计算高度分辨率成像 (X射线微CT) 过程中的成像限制的影响，导致了性质估计中的差异。动力学参数的评估也面临挑战，因为反应系数是关键参数，其数值范围很广。我们解决了这两个问题，并通过将不确定度量化集成到工作流程中，确保了孔隙尺度模型的可靠校准。当前的方法基于反应反问题的多任务公式，将数据驱动和物理建模相结合。

    In this article, we present a novel data assimilation strategy in pore-scale imaging and demonstrate that this makes it possible to robustly address reactive inverse problems incorporating Uncertainty Quantification (UQ). Pore-scale modeling of reactive flow offers a valuable opportunity to investigate the evolution of macro-scale properties subject to dynamic processes. Yet, they suffer from imaging limitations arising from the associated X-ray microtomography (X-ray microCT) process, which induces discrepancies in the properties estimates. Assessment of the kinetic parameters also raises challenges, as reactive coefficients are critical parameters that can cover a wide range of values. We account for these two issues and ensure reliable calibration of pore-scale modeling, based on dynamical microCT images, by integrating uncertainty quantification in the workflow.  The present method is based on a multitasking formulation of reactive inverse problems combining data-driven and physics
    

