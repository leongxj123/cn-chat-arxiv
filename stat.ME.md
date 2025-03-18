# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Logistic-beta processes for modeling dependent random probabilities with beta marginals](https://arxiv.org/abs/2402.07048) | 本文提出了一种新颖的logistic-beta过程用于建模具有beta边际分布的相关随机概率。该过程具有灵活的相关结构和计算优势，并通过非参数二分类回归模拟研究进行了验证。 |
| [^2] | [Simultaneous inference for generalized linear models with unmeasured confounders.](http://arxiv.org/abs/2309.07261) | 本文研究了存在混淆效应时的广义线性模型的大规模假设检验问题，并提出了一种利用正交结构和线性投影的统计估计和推断框架，解决了由于未测混淆因素引起的偏差问题。 |
| [^3] | [Sequential Kernel Embedding for Mediated and Time-Varying Dose Response Curves.](http://arxiv.org/abs/2111.03950) | 本论文提出了一种基于核岭回归的简单非参数估计方法，可以用于估计介导和时变剂量响应曲线。通过引入序贯核嵌入技术，我们实现了对复杂因果估计的简化。通过模拟实验和真实数据的估计结果，证明了该方法的强大性能和普适性。 |

# 详细

[^1]: 用于建模具有beta边际分布的相关随机概率的logistic-beta过程

    Logistic-beta processes for modeling dependent random probabilities with beta marginals

    [https://arxiv.org/abs/2402.07048](https://arxiv.org/abs/2402.07048)

    本文提出了一种新颖的logistic-beta过程用于建模具有beta边际分布的相关随机概率。该过程具有灵活的相关结构和计算优势，并通过非参数二分类回归模拟研究进行了验证。

    

    beta分布被广泛应用于概率建模，并在统计学和机器学习中被广泛使用，尤其在贝叶斯非参数领域。尽管其被广泛使用，但在建模相关随机概率的灵活和计算方便的随机过程扩展方面，相关工作有限。我们提出了一种新颖的随机过程，称为logistic-beta过程，其logistic变换生成具有常见beta边际分布的随机过程。类似于高斯过程，logistic-beta过程可以建模离散和连续域（例如空间或时间）上的相关性，并通过相关核函数具有高度灵活的相关结构。此外，它的正态方差-均值混合表示导致了高效的后验推理算法。通过非参数二分类回归模拟研究，展示了logistic-beta过程的灵活性和计算优势。

    The beta distribution serves as a canonical tool for modeling probabilities and is extensively used in statistics and machine learning, especially in the field of Bayesian nonparametrics. Despite its widespread use, there is limited work on flexible and computationally convenient stochastic process extensions for modeling dependent random probabilities. We propose a novel stochastic process called the logistic-beta process, whose logistic transformation yields a stochastic process with common beta marginals. Similar to the Gaussian process, the logistic-beta process can model dependence on both discrete and continuous domains, such as space or time, and has a highly flexible dependence structure through correlation kernels. Moreover, its normal variance-mean mixture representation leads to highly effective posterior inference algorithms. The flexibility and computational benefits of logistic-beta processes are demonstrated through nonparametric binary regression simulation studies. Fur
    
[^2]: 具有未测混淆因素的广义线性模型的同时推断

    Simultaneous inference for generalized linear models with unmeasured confounders. (arXiv:2309.07261v1 [stat.ME])

    [http://arxiv.org/abs/2309.07261](http://arxiv.org/abs/2309.07261)

    本文研究了存在混淆效应时的广义线性模型的大规模假设检验问题，并提出了一种利用正交结构和线性投影的统计估计和推断框架，解决了由于未测混淆因素引起的偏差问题。

    

    在基因组研究中，常常进行成千上万个同时假设检验，以确定差异表达的基因。然而，由于存在未测混淆因素，许多标准统计方法可能存在严重的偏差。本文研究了存在混淆效应时的多元广义线性模型的大规模假设检验问题。在任意混淆机制下，我们提出了一个统一的统计估计和推断方法，利用正交结构并将线性投影整合到三个关键阶段中。首先，利用多元响应变量分离边际和不相关的混淆效应，恢复混淆系数的列空间。随后，利用$\ell_1$正则化进行稀疏性估计，并强加正交性限制于混淆系数，联合估计潜在因子和主要效应。最后，我们结合投影和加权偏差校正步骤。

    Tens of thousands of simultaneous hypothesis tests are routinely performed in genomic studies to identify differentially expressed genes. However, due to unmeasured confounders, many standard statistical approaches may be substantially biased. This paper investigates the large-scale hypothesis testing problem for multivariate generalized linear models in the presence of confounding effects. Under arbitrary confounding mechanisms, we propose a unified statistical estimation and inference framework that harnesses orthogonal structures and integrates linear projections into three key stages. It first leverages multivariate responses to separate marginal and uncorrelated confounding effects, recovering the confounding coefficients' column space. Subsequently, latent factors and primary effects are jointly estimated, utilizing $\ell_1$-regularization for sparsity while imposing orthogonality onto confounding coefficients. Finally, we incorporate projected and weighted bias-correction steps 
    
[^3]: 序贯核嵌入用于介导和时变剂量响应曲线

    Sequential Kernel Embedding for Mediated and Time-Varying Dose Response Curves. (arXiv:2111.03950v4 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2111.03950](http://arxiv.org/abs/2111.03950)

    本论文提出了一种基于核岭回归的简单非参数估计方法，可以用于估计介导和时变剂量响应曲线。通过引入序贯核嵌入技术，我们实现了对复杂因果估计的简化。通过模拟实验和真实数据的估计结果，证明了该方法的强大性能和普适性。

    

    我们提出了基于核岭回归的介导和时变剂量响应曲线的简单非参数估计器。通过嵌入Pearl的介导公式和Robins的g公式与核函数，我们允许处理、介导者和协变量在一般空间中连续变化，也允许非线性的处理-混淆因素反馈。我们的关键创新是一种称为序贯核嵌入的再生核希尔伯特空间技术，我们使用它来构建复杂因果估计的简单估计器。我们的估计器保留了经典识别的普适性，同时实现了非渐进均匀收敛速度。在具有许多协变量的非线性模拟中，我们展示了强大的性能。我们估计了美国职业训练团的介导和时变剂量响应曲线，并清洁可能成为未来工作基准的数据。我们将我们的结果推广到介导和时变处理效应以及反事实分布，验证了半参数效率。

    We propose simple nonparametric estimators for mediated and time-varying dose response curves based on kernel ridge regression. By embedding Pearl's mediation formula and Robins' g-formula with kernels, we allow treatments, mediators, and covariates to be continuous in general spaces, and also allow for nonlinear treatment-confounder feedback. Our key innovation is a reproducing kernel Hilbert space technique called sequential kernel embedding, which we use to construct simple estimators for complex causal estimands. Our estimators preserve the generality of classic identification while also achieving nonasymptotic uniform rates. In nonlinear simulations with many covariates, we demonstrate strong performance. We estimate mediated and time-varying dose response curves of the US Job Corps, and clean data that may serve as a benchmark in future work. We extend our results to mediated and time-varying treatment effects and counterfactual distributions, verifying semiparametric efficiency 
    

