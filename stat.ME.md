# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Propensity Score Alignment of Unpaired Multimodal Data](https://arxiv.org/abs/2404.01595) | 本文提出了一种解决多模态表示学习中对齐不配对样本挑战的方法，通过估计倾向得分来定义样本之间的距离。 |
| [^2] | [Doubly Robust Inference in Causal Latent Factor Models](https://arxiv.org/abs/2402.11652) | 提出了一种双重稳健的估计量框架，可以在现代数据丰富的环境中估计存在未观察混杂因素下平均处理效应，具有良好的有限样本和渐近性质，并在参数速率下将其误差收敛为零均值高斯分布。 |
| [^3] | [Proximal Causal Inference With Text Data.](http://arxiv.org/abs/2401.06687) | 本论文提出了一种使用文本数据进行近因果推断的方法，通过将文本数据分割并使用零样本模型推断出代理变量，然后应用于近邻 g-formula，从而解决了混淆变量完全未观察到的情况。实验结果表明该方法产生了低偏差的估计值。 |
| [^4] | [A comprehensive framework for multi-fidelity surrogate modeling with noisy data: a gray-box perspective.](http://arxiv.org/abs/2401.06447) | 该论文介绍了一个综合的多保真度代理建模框架，能够将黑盒模型和白盒模型的信息结合起来，并能够处理噪声污染的数据，并估计出无噪声的高保真度函数。 |
| [^5] | [Externally Valid Policy Evaluation Combining Trial and Observational Data.](http://arxiv.org/abs/2310.14763) | 这项研究提出了一种结合试验和观察数据的外部有效策略评估方法，利用试验数据对目标人群上的政策结果进行有效推断，并给出了可验证的评估结果。 |
| [^6] | [On the Computational Complexity of Private High-dimensional Model Selection via the Exponential Mechanism.](http://arxiv.org/abs/2310.07852) | 本文研究了在高维稀疏线性回归模型中的差分隐私模型选择问题。我们使用指数机制进行模型选择，并提出了Metropolis-Hastings算法来克服指数搜索空间的计算复杂性。我们的算法在一定边界条件下能够实现强模型恢复性质，并具有多项式混合时间和近似差分隐私性质。 |
| [^7] | [On the potential benefits of entropic regularization for smoothing Wasserstein estimators.](http://arxiv.org/abs/2210.06934) | 本文研究了熵正则化作为一种平滑方法在Wasserstein估计器中的潜在益处，通过替换最优输运成本的正则化版本来实现。主要发现是熵正则化可以以较低的计算成本达到与未正则化的Wasserstein估计器相当的统计性能。 |

# 详细

[^1]: 多模态数据无配对倾向得分对齐

    Propensity Score Alignment of Unpaired Multimodal Data

    [https://arxiv.org/abs/2404.01595](https://arxiv.org/abs/2404.01595)

    本文提出了一种解决多模态表示学习中对齐不配对样本挑战的方法，通过估计倾向得分来定义样本之间的距离。

    

    多模态表示学习技术通常依赖于配对样本来学习共同的表示，但在生物学等领域，往往难以收集配对样本，因为测量设备通常会破坏样本。本文介绍了一种解决多模态表示学习中对齐不配对样本的方法。我们将因果推断中的潜在结果与多模态观察中的潜在视图进行类比，这使我们能够使用Rubin的框架来估计一个共同的空间，以匹配样本。我们的方法假设我们收集了经过处理实验干扰的样本，并利用此来从每种模态中估计倾向得分，其中包括潜在状态和处理之间的所有共享信息，并可用于定义样本之间的距离。我们尝试了两种利用这一方法的对齐技术。

    arXiv:2404.01595v1 Announce Type: new  Abstract: Multimodal representation learning techniques typically rely on paired samples to learn common representations, but paired samples are challenging to collect in fields such as biology where measurement devices often destroy the samples. This paper presents an approach to address the challenge of aligning unpaired samples across disparate modalities in multimodal representation learning. We draw an analogy between potential outcomes in causal inference and potential views in multimodal observations, which allows us to use Rubin's framework to estimate a common space in which to match samples. Our approach assumes we collect samples that are experimentally perturbed by treatments, and uses this to estimate a propensity score from each modality, which encapsulates all shared information between a latent state and treatment and can be used to define a distance between samples. We experiment with two alignment techniques that leverage this di
    
[^2]: 因果潜在因子模型中的双重稳健推断

    Doubly Robust Inference in Causal Latent Factor Models

    [https://arxiv.org/abs/2402.11652](https://arxiv.org/abs/2402.11652)

    提出了一种双重稳健的估计量框架，可以在现代数据丰富的环境中估计存在未观察混杂因素下平均处理效应，具有良好的有限样本和渐近性质，并在参数速率下将其误差收敛为零均值高斯分布。

    

    本文介绍了一种在现代数据丰富环境中估计存在未观察混杂因素下的平均处理效应的新框架，该环境具有大量单位和结果。所提出的估计量是双重稳健的，结合了结果填补、倒数概率加权以及一种用于矩阵补全的新型交叉配对程序。我们推导了有限样本和渐近保证，并展示了新估计量的误差收敛到参数速率下的零均值高斯分布。模拟结果展示了本文分析的估计量的形式特性的实际相关性。

    arXiv:2402.11652v1 Announce Type: cross  Abstract: This article introduces a new framework for estimating average treatment effects under unobserved confounding in modern data-rich environments featuring large numbers of units and outcomes. The proposed estimator is doubly robust, combining outcome imputation, inverse probability weighting, and a novel cross-fitting procedure for matrix completion. We derive finite-sample and asymptotic guarantees, and show that the error of the new estimator converges to a mean-zero Gaussian distribution at a parametric rate. Simulation results demonstrate the practical relevance of the formal properties of the estimators analyzed in this article.
    
[^3]: 使用文本数据的近因果推断

    Proximal Causal Inference With Text Data. (arXiv:2401.06687v1 [cs.CL])

    [http://arxiv.org/abs/2401.06687](http://arxiv.org/abs/2401.06687)

    本论文提出了一种使用文本数据进行近因果推断的方法，通过将文本数据分割并使用零样本模型推断出代理变量，然后应用于近邻 g-formula，从而解决了混淆变量完全未观察到的情况。实验结果表明该方法产生了低偏差的估计值。

    

    最近的基于文本的因果方法试图通过将非结构化文本数据作为倾向于包含部分或不完全测量的混淆变量的代理来减轻混淆偏差。这些方法假设分析人员在一部分实例的文本中具有有监督的混淆变量标签，但由于数据隐私或成本，这种约束并不总是可行。在这里，我们解决了一个重要的混淆变量完全未观察到的情况。我们提出了一种新的因果推断方法，将处理前文本数据分割，并使用两个零样本模型从分割的两个部分推断出两个代理，并将这些代理应用于近邻 g-formula。我们证明了我们基于文本的代理方法满足近邻 g-formula所需的识别条件，而其他看似合理的提议则不满足。我们在合成和半合成环境中评估了我们的方法，并发现它产生了低偏差的估计值。

    Recent text-based causal methods attempt to mitigate confounding bias by including unstructured text data as proxies of confounding variables that are partially or imperfectly measured. These approaches assume analysts have supervised labels of the confounders given text for a subset of instances, a constraint that is not always feasible due to data privacy or cost. Here, we address settings in which an important confounding variable is completely unobserved. We propose a new causal inference method that splits pre-treatment text data, infers two proxies from two zero-shot models on the separate splits, and applies these proxies in the proximal g-formula. We prove that our text-based proxy method satisfies identification conditions required by the proximal g-formula while other seemingly reasonable proposals do not. We evaluate our method in synthetic and semi-synthetic settings and find that it produces estimates with low bias. This combination of proximal causal inference and zero-sh
    
[^4]: 一个综合的多保真度代理建模框架，带有噪声数据：从灰盒的角度来看

    A comprehensive framework for multi-fidelity surrogate modeling with noisy data: a gray-box perspective. (arXiv:2401.06447v1 [stat.ME])

    [http://arxiv.org/abs/2401.06447](http://arxiv.org/abs/2401.06447)

    该论文介绍了一个综合的多保真度代理建模框架，能够将黑盒模型和白盒模型的信息结合起来，并能够处理噪声污染的数据，并估计出无噪声的高保真度函数。

    

    计算机模拟（即白盒模型）在模拟复杂工程系统方面比以往任何时候都更加必不可少。然而，仅凭计算模型往往无法完全捕捉现实的复杂性。当物理实验可行时，增强计算模型提供的不完整信息变得非常重要。灰盒建模涉及到将数据驱动模型（即黑盒模型）和白盒模型（即基于物理的模型）的信息融合的问题。在本文中，我们提出使用多保真度代理模型（MFSMs）来执行这个任务。MFSM将不同计算保真度的模型的信息集成到一个新的代理模型中。我们提出的多保真度代理建模框架能够处理被噪声污染的数据，并能够估计底层无噪声的高保真度函数。我们的方法强调以置信度的形式提供其预测中不确定性的精确估计。

    Computer simulations (a.k.a. white-box models) are more indispensable than ever to model intricate engineering systems. However, computational models alone often fail to fully capture the complexities of reality. When physical experiments are accessible though, it is of interest to enhance the incomplete information offered by computational models. Gray-box modeling is concerned with the problem of merging information from data-driven (a.k.a. black-box) models and white-box (i.e., physics-based) models. In this paper, we propose to perform this task by using multi-fidelity surrogate models (MFSMs). A MFSM integrates information from models with varying computational fidelity into a new surrogate model. The multi-fidelity surrogate modeling framework we propose handles noise-contaminated data and is able to estimate the underlying noise-free high-fidelity function. Our methodology emphasizes on delivering precise estimates of the uncertainty in its predictions in the form of confidence 
    
[^5]: 外部验证策略评估结合试验和观察数据

    Externally Valid Policy Evaluation Combining Trial and Observational Data. (arXiv:2310.14763v1 [stat.ME])

    [http://arxiv.org/abs/2310.14763](http://arxiv.org/abs/2310.14763)

    这项研究提出了一种结合试验和观察数据的外部有效策略评估方法，利用试验数据对目标人群上的政策结果进行有效推断，并给出了可验证的评估结果。

    

    随机试验被广泛认为是评估决策策略影响的金 standard。然而，试验数据来自可能与目标人群不同的人群，这引发了外部效度（也称为泛化能力）的问题。在本文中，我们试图利用试验数据对目标人群上的政策结果进行有效推断。目标人群的额外协变量数据用于模拟试验研究中个体的抽样。我们开发了一种方法，在任何指定的模型未校准范围内产生可验证的基于试验的政策评估。该方法是非参数的，即使样本是有限的，有效性也得到保证。使用模拟和实际数据说明了认证的政策评估结果。

    Randomized trials are widely considered as the gold standard for evaluating the effects of decision policies. Trial data is, however, drawn from a population which may differ from the intended target population and this raises a problem of external validity (aka. generalizability). In this paper we seek to use trial data to draw valid inferences about the outcome of a policy on the target population. Additional covariate data from the target population is used to model the sampling of individuals in the trial study. We develop a method that yields certifiably valid trial-based policy evaluations under any specified range of model miscalibrations. The method is nonparametric and the validity is assured even with finite samples. The certified policy evaluations are illustrated using both simulated and real data.
    
[^6]: 关于通过指数机制进行高维私有模型选择的计算复杂性

    On the Computational Complexity of Private High-dimensional Model Selection via the Exponential Mechanism. (arXiv:2310.07852v1 [stat.ML])

    [http://arxiv.org/abs/2310.07852](http://arxiv.org/abs/2310.07852)

    本文研究了在高维稀疏线性回归模型中的差分隐私模型选择问题。我们使用指数机制进行模型选择，并提出了Metropolis-Hastings算法来克服指数搜索空间的计算复杂性。我们的算法在一定边界条件下能够实现强模型恢复性质，并具有多项式混合时间和近似差分隐私性质。

    

    在差分隐私框架下，我们考虑了高维稀疏线性回归模型中的模型选择问题。具体而言，我们考虑了差分隐私最佳子集选择的问题，并研究了其效用保证。我们采用了广为人知的指数机制来选择最佳模型，并在一定边界条件下，建立了其强模型恢复性质。然而，指数机制的指数搜索空间导致了严重的计算瓶颈。为了克服这个挑战，我们提出了Metropolis-Hastings算法来进行采样步骤，并在问题参数$n$、$p$和$s$中建立了其到稳态分布的多项式混合时间。此外，我们还利用其混合性质建立了Metropolis-Hastings随机行走的最终估计的近似差分隐私性质。最后，我们还进行了一些说明性模拟，印证了我们主要结果的理论发现。

    We consider the problem of model selection in a high-dimensional sparse linear regression model under the differential privacy framework. In particular, we consider the problem of differentially private best subset selection and study its utility guarantee. We adopt the well-known exponential mechanism for selecting the best model, and under a certain margin condition, we establish its strong model recovery property. However, the exponential search space of the exponential mechanism poses a serious computational bottleneck. To overcome this challenge, we propose a Metropolis-Hastings algorithm for the sampling step and establish its polynomial mixing time to its stationary distribution in the problem parameters $n,p$, and $s$. Furthermore, we also establish approximate differential privacy for the final estimates of the Metropolis-Hastings random walk using its mixing property. Finally, we also perform some illustrative simulations that echo the theoretical findings of our main results
    
[^7]: 关于使用熵正则化平滑Wasserstein估计器的潜在益处

    On the potential benefits of entropic regularization for smoothing Wasserstein estimators. (arXiv:2210.06934v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2210.06934](http://arxiv.org/abs/2210.06934)

    本文研究了熵正则化作为一种平滑方法在Wasserstein估计器中的潜在益处，通过替换最优输运成本的正则化版本来实现。主要发现是熵正则化可以以较低的计算成本达到与未正则化的Wasserstein估计器相当的统计性能。

    

    本文专注于研究熵正则化在最优输运中作为Wasserstein估计器的平滑方法，通过统计学中逼近误差和估计误差的经典权衡。Wasserstein估计器被定义为解决变分问题的解，其目标函数涉及概率测度之间的最优输运成本的使用。这样的估计器可以通过用熵惩罚替换最优输运成本的正则化版本来进行正则化，从而对结果估计器产生潜在的平滑效果。在这项工作中，我们探讨了熵正则化对正则化Wasserstein估计器的逼近和估计性质可能带来的益处。我们的主要贡献是讨论熵正则化如何以更低的计算成本达到与未正则化的Wasserstein估计器相当的统计性能。

    This paper is focused on the study of entropic regularization in optimal transport as a smoothing method for Wasserstein estimators, through the prism of the classical tradeoff between approximation and estimation errors in statistics. Wasserstein estimators are defined as solutions of variational problems whose objective function involves the use of an optimal transport cost between probability measures. Such estimators can be regularized by replacing the optimal transport cost by its regularized version using an entropy penalty on the transport plan. The use of such a regularization has a potentially significant smoothing effect on the resulting estimators. In this work, we investigate its potential benefits on the approximation and estimation properties of regularized Wasserstein estimators. Our main contribution is to discuss how entropic regularization may reach, at a lower computational cost, statistical performances that are comparable to those of un-regularized Wasserstein esti
    

