# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Iterative Methods for Vecchia-Laplace Approximations for Latent Gaussian Process Models.](http://arxiv.org/abs/2310.12000) | 这篇文章介绍了用于潜在高斯过程模型中的Vecchia-Laplace近似法的迭代方法，相比于传统的Cholesky分解方法，可以显著加快计算速度。 |
| [^2] | [Filtered and Unfiltered Treatment Effects with Targeting Instruments.](http://arxiv.org/abs/2007.10432) | 本文研究如何使用有目标工具来控制多值处理中的选择偏差，并建立了组合编译器群体的条件来确定反事实平均值和处理效果。 |

# 详细

[^1]: Vecchia-Laplace近似法在潜在高斯过程模型中的迭代方法

    Iterative Methods for Vecchia-Laplace Approximations for Latent Gaussian Process Models. (arXiv:2310.12000v1 [stat.ME])

    [http://arxiv.org/abs/2310.12000](http://arxiv.org/abs/2310.12000)

    这篇文章介绍了用于潜在高斯过程模型中的Vecchia-Laplace近似法的迭代方法，相比于传统的Cholesky分解方法，可以显著加快计算速度。

    

    潜在高斯过程（GP）模型是灵活的概率非参数函数模型。Vecchia近似是用于克服大数据计算瓶颈的准确近似方法，Laplace近似是一种快速方法，可以近似非高斯似然函数的边缘似然和后验预测分布，并具有渐近收敛保证。然而，当与直接求解方法（如Cholesky分解）结合使用时，Vecchia-Laplace近似的计算复杂度增长超线性地随样本大小增加。因此，与Vecchia-Laplace近似计算相关的运算在通常情况下是最准确的大型数据集时会变得非常缓慢。在本文中，我们提出了几种用于Vecchia-Laplace近似推断的迭代方法，相比于基于Cholesky的计算，可以大大加快计算速度。我们对我们的方法进行了分析。

    Latent Gaussian process (GP) models are flexible probabilistic non-parametric function models. Vecchia approximations are accurate approximations for GPs to overcome computational bottlenecks for large data, and the Laplace approximation is a fast method with asymptotic convergence guarantees to approximate marginal likelihoods and posterior predictive distributions for non-Gaussian likelihoods. Unfortunately, the computational complexity of combined Vecchia-Laplace approximations grows faster than linearly in the sample size when used in combination with direct solver methods such as the Cholesky decomposition. Computations with Vecchia-Laplace approximations thus become prohibitively slow precisely when the approximations are usually the most accurate, i.e., on large data sets. In this article, we present several iterative methods for inference with Vecchia-Laplace approximations which make computations considerably faster compared to Cholesky-based calculations. We analyze our propo
    
[^2]: 有目标工具的过滤与未过滤处理效果

    Filtered and Unfiltered Treatment Effects with Targeting Instruments. (arXiv:2007.10432v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2007.10432](http://arxiv.org/abs/2007.10432)

    本文研究如何使用有目标工具来控制多值处理中的选择偏差，并建立了组合编译器群体的条件来确定反事实平均值和处理效果。

    

    在应用中，多值处理是很常见的。我们探讨了在这种情况下使用离散工具来控制选择偏差的方法。我们强调了有关定位（工具定位于哪些处理）和过滤（限制分析师对给定观测的处理分配的知识）的假设作用。这允许我们建立条件，使得针对组合编译器群体，可以确定反事实平均值和处理效果。我们通过将其应用于Head Start Impact Study和Student Achievement and Retention Project的数据来说明我们框架的实用性。

    Multivalued treatments are commonplace in applications. We explore the use of discrete-valued instruments to control for selection bias in this setting. Our discussion stresses the role of assumptions on targeting (which instruments target which treatments) and filtering (limits on the analyst's knowledge of the treatment assigned to a given observation). It allows us to establish conditions under which counterfactual averages and treatment effects are identified for composite complier groups. We illustrate the usefulness of our framework by applying it to data from the Head Start Impact Study and the Student Achievement and Retention Project.
    

