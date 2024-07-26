# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Priori Uncertainty Quantification of Reacting Turbulence Closure Models using Bayesian Neural Networks](https://arxiv.org/abs/2402.18729) | 使用贝叶斯神经网络对反应流动模型中的不确定性进行量化，特别是在湍涡预混火焰动态中关键变量的建模方面取得重要进展 |
| [^2] | [Generalizable improvement of the Spalart-Allmaras model through assimilation of experimental data.](http://arxiv.org/abs/2309.06679) | 本研究通过实验数据同化改进了Spalart-Allmaras模型，实现了对分离流体的雷诺平均纳维-斯托克斯解的泛化，提高了计算模型的性能。 |
| [^3] | [The most likely common cause.](http://arxiv.org/abs/2306.17557) | 对于因果不充分的情况下的共同原因问题，我们使用广义最大似然方法来识别共同原因C，与最大熵原则密切相关。对于两个二元对称变量的研究揭示了类似于二阶相变的条件概率非解析行为。 |

# 详细

[^1]: 利用贝叶斯神经网络对反应湍流封闭模型进行先验不确定性量化

    A Priori Uncertainty Quantification of Reacting Turbulence Closure Models using Bayesian Neural Networks

    [https://arxiv.org/abs/2402.18729](https://arxiv.org/abs/2402.18729)

    使用贝叶斯神经网络对反应流动模型中的不确定性进行量化，特别是在湍涡预混火焰动态中关键变量的建模方面取得重要进展

    

    虽然为大涡模拟（LES）中的子滤波尺度（SFS）提出了许多基于物理的封闭模型形式，但直接数值模拟（DNS）提供的大量数据为利用数据驱动建模技术创造了机会。尽管灵活，数据驱动模型仍取决于选择的数据集和模型的函数形式。采用这种模型的增加需要可靠地估计数据驱动模型中数据知识和超出分布范围的不确定性。在本工作中，我们利用贝叶斯神经网络（BNNs）来捕捉反应流动模型中的逻辑不确定性和偶然不确定性。特别是，我们模拟了在湍涡预混火焰动态中起关键作用的滤波进展变量标量耗散率。我们展示了BNN模型可以提供关于数据驱动封闭模型的不确定性结构的独特见解。我们还提出了一种方法来进行...

    arXiv:2402.18729v1 Announce Type: cross  Abstract: While many physics-based closure model forms have been posited for the sub-filter scale (SFS) in large eddy simulation (LES), vast amounts of data available from direct numerical simulation (DNS) create opportunities to leverage data-driven modeling techniques. Albeit flexible, data-driven models still depend on the dataset and the functional form of the model chosen. Increased adoption of such models requires reliable uncertainty estimates both in the data-informed and out-of-distribution regimes. In this work, we employ Bayesian neural networks (BNNs) to capture both epistemic and aleatoric uncertainties in a reacting flow model. In particular, we model the filtered progress variable scalar dissipation rate which plays a key role in the dynamics of turbulent premixed flames. We demonstrate that BNN models can provide unique insights about the structure of uncertainty of the data-driven closure models. We also propose a method for the
    
[^2]: 通过实验数据同化实现对Spalart-Allmaras模型的可普适改进

    Generalizable improvement of the Spalart-Allmaras model through assimilation of experimental data. (arXiv:2309.06679v1 [physics.flu-dyn])

    [http://arxiv.org/abs/2309.06679](http://arxiv.org/abs/2309.06679)

    本研究通过实验数据同化改进了Spalart-Allmaras模型，实现了对分离流体的雷诺平均纳维-斯托克斯解的泛化，提高了计算模型的性能。

    

    本研究旨在利用模型和数据融合改进分离流体的雷诺平均纳维-斯托克斯解的Spalart-Allmaras（SA）闭合模型。特别是，我们的目标是开发模型，不仅能将稀疏的实验数据同化以改善计算模型的性能，还能通过恢复经典的SA行为来推广到未见过的情况。我们使用数据同化，即集合卡尔曼滤波方法（EnKF），通过将SA模型的系数校准到分离流体中来实现我们的目标。通过参数化产生、扩散和破坏项，实现了一种全面的校准策略。该校准依赖于采集的分离流体速度剖面、壁擦力和压力系数的实验数据的同化。尽管仅使用了来自单一流动条件（环绕一个背面台阶）的观测数据，但重新校准的SA模型表现出泛化能力。

    This study focuses on the use of model and data fusion for improving the Spalart-Allmaras (SA) closure model for Reynolds-averaged Navier-Stokes solutions of separated flows. In particular, our goal is to develop of models that not-only assimilate sparse experimental data to improve performance in computational models, but also generalize to unseen cases by recovering classical SA behavior. We achieve our goals using data assimilation, namely the Ensemble Kalman Filtering approach (EnKF), to calibrate the coefficients of the SA model for separated flows. A holistic calibration strategy is implemented via a parameterization of the production, diffusion, and destruction terms. This calibration relies on the assimilation of experimental data collected velocity profiles, skin friction, and pressure coefficients for separated flows. Despite using of observational data from a single flow condition around a backward-facing step (BFS), the recalibrated SA model demonstrates generalization to o
    
[^3]: 最可能的共同原因

    The most likely common cause. (arXiv:2306.17557v1 [physics.data-an])

    [http://arxiv.org/abs/2306.17557](http://arxiv.org/abs/2306.17557)

    对于因果不充分的情况下的共同原因问题，我们使用广义最大似然方法来识别共同原因C，与最大熵原则密切相关。对于两个二元对称变量的研究揭示了类似于二阶相变的条件概率非解析行为。

    

    对于两个随机变量A和B的共同原因原则在因果不充分的情况下进行了研究，当它们的共同原因C被认为已经存在，但只观测到了A和B的联合概率。因此，C不能被唯一确定（潜在混杂因子问题）。我们展示了广义最大似然方法可以应用于这种情况，并且允许识别与共同原因原则一致的C。它与最大熵原则密切相关。对两个二元对称变量的研究揭示了条件概率的非解析行为，类似于二阶相变。这发生在观察到的概率分布从相关到反相关的过渡期间。讨论了广义似然方法与其他方法（如预测似然和最小共同原因熵）之间的关系。

    The common cause principle for two random variables $A$ and $B$ is examined in the case of causal insufficiency, when their common cause $C$ is known to exist, but only the joint probability of $A$ and $B$ is observed. As a result, $C$ cannot be uniquely identified (the latent confounder problem). We show that the generalized maximum likelihood method can be applied to this situation and allows identification of $C$ that is consistent with the common cause principle. It closely relates to the maximum entropy principle. Investigation of the two binary symmetric variables reveals a non-analytic behavior of conditional probabilities reminiscent of a second-order phase transition. This occurs during the transition from correlation to anti-correlation in the observed probability distribution. The relation between the generalized likelihood approach and alternative methods, such as predictive likelihood and the minimum common cause entropy, is discussed. The consideration of the common cause
    

