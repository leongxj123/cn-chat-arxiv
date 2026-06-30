# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Large (and Deep) Factor Models](https://arxiv.org/abs/2402.06635) | 本文通过证明一个足够宽而任意深的神经网络训练出来的投资组合优化模型与大型因子模型等效，打开了深度学习在此领域中的黑盒子，并提供了一种封闭形式的推导方法。研究实证了不同架构选择对模型性能的影响，并证明了随着深度增加，模型在足够多数据下的表现逐渐提升，直至达到饱和。 |
| [^2] | [Multiply Robust Causal Mediation Analysis with Continuous Treatments](https://arxiv.org/abs/2105.09254) | 本文提出了一种适用于连续治疗环境的多重稳健因果中介分析估计器，采用了核平滑方法，并具有多重稳健性和渐近正态性。 |
| [^3] | [Granular ball computing: an efficient, robust, and interpretable adaptive multi-granularity representation and computation method.](http://arxiv.org/abs/2304.11171) | 本文提出了一种基于颗粒球计算的自适应多粒度表示和计算方法，能够提高机器学习的效率、鲁棒性和可解释性。 |

# 详细

[^1]: 大型（和深度）因子模型

    Large (and Deep) Factor Models

    [https://arxiv.org/abs/2402.06635](https://arxiv.org/abs/2402.06635)

    本文通过证明一个足够宽而任意深的神经网络训练出来的投资组合优化模型与大型因子模型等效，打开了深度学习在此领域中的黑盒子，并提供了一种封闭形式的推导方法。研究实证了不同架构选择对模型性能的影响，并证明了随着深度增加，模型在足够多数据下的表现逐渐提升，直至达到饱和。

    

    我们打开了深度学习在投资组合优化中的黑盒子，并证明了一个足够宽而任意深的神经网络(DNN)被训练用来最大化随机贴现因子(SDF)的夏普比率等效于一个大型因子模型(LFM)：一个使用许多非线性特征的线性因子定价模型。这些特征的性质取决于DNN的体系结构，在一种明确可追踪的方式下。这使得首次可以推导出封闭形式的端到端训练的基于DNN的SDF。我们通过实证评估了LFMs，并展示了各种架构选择如何影响SDF的性能。我们证明了深度复杂性的优点：随着足够多的数据，DNN-SDF的外样总体表现会随着神经网络的深度而增加，当隐藏层达到约100层时达到饱和。

    We open up the black box behind Deep Learning for portfolio optimization and prove that a sufficiently wide and arbitrarily deep neural network (DNN) trained to maximize the Sharpe ratio of the Stochastic Discount Factor (SDF) is equivalent to a large factor model (LFM): A linear factor pricing model that uses many non-linear characteristics. The nature of these characteristics depends on the architecture of the DNN in an explicit, tractable fashion. This makes it possible to derive end-to-end trained DNN-based SDFs in closed form for the first time. We evaluate LFMs empirically and show how various architectural choices impact SDF performance. We document the virtue of depth complexity: With enough data, the out-of-sample performance of DNN-SDF is increasing in the NN depth, saturating at huge depths of around 100 hidden layers.
    
[^2]: 在连续治疗下的多重稳健因果中介分析

    Multiply Robust Causal Mediation Analysis with Continuous Treatments

    [https://arxiv.org/abs/2105.09254](https://arxiv.org/abs/2105.09254)

    本文提出了一种适用于连续治疗环境的多重稳健因果中介分析估计器，采用了核平滑方法，并具有多重稳健性和渐近正态性。

    

    在许多应用中，研究人员对治疗或暴露对感兴趣的结果的直接和间接的因果效应。中介分析为鉴定和估计这些因果效应提供了一个严谨的框架。对于二元治疗，Tchetgen Tchetgen和Shpitser (2012)提出了直接和间接效应的高效估计器，基于参数的影响函数。这些估计器具有良好的性质，如多重稳健性和渐近正态性，同时允许对干扰参数进行低于根号n的收敛速度。然而，在涉及连续治疗的情况下，这些基于影响函数的估计器没有准备好应用，除非进行强参数假设。在这项工作中，我们利用核平滑方法提出了一种适用于连续治疗环境的估计器，受到Tchetgen Tchetgen的影响函数估计器的启发。

    In many applications, researchers are interested in the direct and indirect causal effects of a treatment or exposure on an outcome of interest. Mediation analysis offers a rigorous framework for identifying and estimating these causal effects. For binary treatments, efficient estimators for the direct and indirect effects are presented in Tchetgen Tchetgen and Shpitser (2012) based on the influence function of the parameter of interest. These estimators possess desirable properties, such as multiple-robustness and asymptotic normality, while allowing for slower than root-n rates of convergence for the nuisance parameters. However, in settings involving continuous treatments, these influence function-based estimators are not readily applicable without making strong parametric assumptions. In this work, utilizing a kernel-smoothing approach, we propose an estimator suitable for settings with continuous treatments inspired by the influence function-based estimator of Tchetgen Tchetgen an
    
[^3]: 颗粒球计算：一种高效、鲁棒和可解释的自适应多粒度表示和计算方法

    Granular ball computing: an efficient, robust, and interpretable adaptive multi-granularity representation and computation method. (arXiv:2304.11171v1 [cs.LG])

    [http://arxiv.org/abs/2304.11171](http://arxiv.org/abs/2304.11171)

    本文提出了一种基于颗粒球计算的自适应多粒度表示和计算方法，能够提高机器学习的效率、鲁棒性和可解释性。

    

    人类认知具有“先大后小”的认知机制，因此具有自适应的多粒度描述能力。这导致了有效性、鲁棒性和可解释性等计算特性。本文提出了一种新的基于颗粒球计算的自适应多粒度表示和计算方法。他们将这种方法应用于几个机器学习任务，并证明其相对于其他最先进的方法的有效性。

    Human cognition has a ``large-scale first'' cognitive mechanism, therefore possesses adaptive multi-granularity description capabilities. This results in computational characteristics such as efficiency, robustness, and interpretability. Although most existing artificial intelligence learning methods have certain multi-granularity features, they do not fully align with the ``large-scale first'' cognitive mechanism. Multi-granularity granular-ball computing is an important model method developed in recent years. This method can use granular-balls of different sizes to adaptively represent and cover the sample space, and perform learning based on granular-balls. Since the number of coarse-grained "granular-ball" is smaller than the number of sample points, granular-ball computing is more efficient; the coarse-grained characteristics of granular-balls are less likely to be affected by fine-grained sample points, making them more robust; the multi-granularity structure of granular-balls ca
    

