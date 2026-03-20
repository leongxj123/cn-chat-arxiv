# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Combining T-learning and DR-learning: a framework for oracle-efficient estimation of causal contrasts](https://arxiv.org/abs/2402.01972) | 这篇论文介绍了高效插件学习的框架，能够有效估计异质因果对比，并解决了其他学习策略的一些缺点。该框架构建了人口风险函数的高效插件估计器，具有稳定性和鲁棒性。 |
| [^2] | [Adaptive debiased machine learning using data-driven model selection techniques.](http://arxiv.org/abs/2307.12544) | 提出了一种自适应去偏机器学习（ADML）框架，通过结合数据驱动的模型选择和去偏机器学习技术，构建了渐进线性、自适应和超效率的路径可微的功能估计器。 |

# 详细

[^1]: 组合T-learning和DR-learning：一个用于高效估计因果对比的框架

    Combining T-learning and DR-learning: a framework for oracle-efficient estimation of causal contrasts

    [https://arxiv.org/abs/2402.01972](https://arxiv.org/abs/2402.01972)

    这篇论文介绍了高效插件学习的框架，能够有效估计异质因果对比，并解决了其他学习策略的一些缺点。该框架构建了人口风险函数的高效插件估计器，具有稳定性和鲁棒性。

    

    我们引入了高效插件（EP）学习，这是一种用于估计异质因果对比的新框架，例如条件平均处理效应和条件相对风险。 EP学习框架享有与Neyman正交学习策略（如DR-learning和R-learning）相同的oracle效率，同时解决了它们的一些主要缺点，包括（i）实际适用性可能受到损失函数非凸性的阻碍； （ii）它们可能因违反界限的倒数概率加权和伪结果而导致性能和稳定性差。为了避免这些缺点，EP学习者构建了因果对比的人口风险函数的高效插件估计器，从而继承了T-learning等插件估计策略的稳定性和鲁棒性特性。在合理条件下，基于经验风险最小化的EP学习者具有oracle效率，表现出渐近等价的性质。

    We introduce efficient plug-in (EP) learning, a novel framework for the estimation of heterogeneous causal contrasts, such as the conditional average treatment effect and conditional relative risk. The EP-learning framework enjoys the same oracle-efficiency as Neyman-orthogonal learning strategies, such as DR-learning and R-learning, while addressing some of their primary drawbacks, including that (i) their practical applicability can be hindered by loss function non-convexity; and (ii) they may suffer from poor performance and instability due to inverse probability weighting and pseudo-outcomes that violate bounds. To avoid these drawbacks, EP-learner constructs an efficient plug-in estimator of the population risk function for the causal contrast, thereby inheriting the stability and robustness properties of plug-in estimation strategies like T-learning. Under reasonable conditions, EP-learners based on empirical risk minimization are oracle-efficient, exhibiting asymptotic equivalen
    
[^2]: 自适应去偏机器学习方法及数据驱动模型选择技术

    Adaptive debiased machine learning using data-driven model selection techniques. (arXiv:2307.12544v1 [stat.ME])

    [http://arxiv.org/abs/2307.12544](http://arxiv.org/abs/2307.12544)

    提出了一种自适应去偏机器学习（ADML）框架，通过结合数据驱动的模型选择和去偏机器学习技术，构建了渐进线性、自适应和超效率的路径可微的功能估计器。

    

    非参数推断中的去偏机器学习估计器可能存在过高的变异性和不稳定性。为了解决这个问题，我们提出了自适应去偏机器学习（ADML）框架，通过结合数据驱动的模型选择和去偏机器学习技术，构建渐进线性、自适应和超效率的路径可微的功能估计器。通过从数据中直接学习模型结构，ADML避免了模型规范错误引入的偏差，并摆脱了参数化和半参数化模型的限制。

    Debiased machine learning estimators for nonparametric inference of smooth functionals of the data-generating distribution can suffer from excessive variability and instability. For this reason, practitioners may resort to simpler models based on parametric or semiparametric assumptions. However, such simplifying assumptions may fail to hold, and estimates may then be biased due to model misspecification. To address this problem, we propose Adaptive Debiased Machine Learning (ADML), a nonparametric framework that combines data-driven model selection and debiased machine learning techniques to construct asymptotically linear, adaptive, and superefficient estimators for pathwise differentiable functionals. By learning model structure directly from data, ADML avoids the bias introduced by model misspecification and remains free from the restrictions of parametric and semiparametric models. While they may exhibit irregular behavior for the target parameter in a nonparametric statistical mo
    

