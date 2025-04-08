# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Differentially Private Sliced Inverse Regression: Minimax Optimality and Algorithm.](http://arxiv.org/abs/2401.08150) | 本文提出了针对充足维度减少中的隐私问题的最佳差分隐私算法，并在低维和高维设置下建立了不同ially private 切片逆回归的下界。通过仿真和真实数据分析验证了这些算法的有效性。 |
| [^2] | [Reparametrization and the Semiparametric Bernstein-von-Mises Theorem.](http://arxiv.org/abs/2306.03816) | 本文提出了一种参数化形式，该形式可以通过生成Neyman正交矩条件来降低对干扰参数的敏感度，从而可以用于去偏贝叶斯推断中的后验分布，同时在参数速率下对低维参数进行真实值的收缩，并在半参数效率界的方差下进行渐近正态分布。 |

# 详细

[^1]: 差分隐私切片逆回归: 极小极大性和算法

    Differentially Private Sliced Inverse Regression: Minimax Optimality and Algorithm. (arXiv:2401.08150v1 [stat.ML])

    [http://arxiv.org/abs/2401.08150](http://arxiv.org/abs/2401.08150)

    本文提出了针对充足维度减少中的隐私问题的最佳差分隐私算法，并在低维和高维设置下建立了不同ially private 切片逆回归的下界。通过仿真和真实数据分析验证了这些算法的有效性。

    

    随着数据驱动应用的普及，隐私保护已成为高维数据分析中的一个关键问题。切片逆回归是一种广泛应用的统计技术，通过降低协变量的维度，同时保持足够的统计信息。本文提出了针对充足维度减少中的隐私问题的最佳差分隐私算法。我们在低维和高维设置下建立了不同ially private 切片逆回归的下界。此外，我们设计了差分隐私算法，实现了极小极大下界的要求，并在降维空间中同时保护隐私和保存重要信息的有效性。通过一系列的仿真实验和真实数据分析，我们证明了这些差分隐私算法的有效性。

    Privacy preservation has become a critical concern in high-dimensional data analysis due to the growing prevalence of data-driven applications. Proposed by Li (1991), sliced inverse regression has emerged as a widely utilized statistical technique for reducing covariate dimensionality while maintaining sufficient statistical information. In this paper, we propose optimally differentially private algorithms specifically designed to address privacy concerns in the context of sufficient dimension reduction. We proceed to establish lower bounds for differentially private sliced inverse regression in both the low and high-dimensional settings. Moreover, we develop differentially private algorithms that achieve the minimax lower bounds up to logarithmic factors. Through a combination of simulations and real data analysis, we illustrate the efficacy of these differentially private algorithms in safeguarding privacy while preserving vital information within the reduced dimension space. As a na
    
[^2]: 重参数化与半参数Bernstein-von-Mises定理

    Reparametrization and the Semiparametric Bernstein-von-Mises Theorem. (arXiv:2306.03816v1 [math.ST])

    [http://arxiv.org/abs/2306.03816](http://arxiv.org/abs/2306.03816)

    本文提出了一种参数化形式，该形式可以通过生成Neyman正交矩条件来降低对干扰参数的敏感度，从而可以用于去偏贝叶斯推断中的后验分布，同时在参数速率下对低维参数进行真实值的收缩，并在半参数效率界的方差下进行渐近正态分布。

    

    本文考虑了部分线性模型的贝叶斯推断。我们的方法利用了回归函数的一个参数化形式，该形式专门用于估计所关心的低维参数。参数化的关键特性是生成了一个Neyman正交矩条件，这意味着对干扰参数的估计低维参数不太敏感。我们的大样本分析支持了这种说法。特别地，我们推导出充分的条件，使得低维参数的后验在参数速率下对真实值收缩，并且在半参数效率界的方差下渐近地正态分布。这些条件相对于回归模型的原始参数化允许更大类的干扰参数。总的来说，我们得出结论，一个嵌入了Neyman正交性的参数化方法可以成为半参数推断中的一个有用工具，以去偏后验分布。

    This paper considers Bayesian inference for the partially linear model. Our approach exploits a parametrization of the regression function that is tailored toward estimating a low-dimensional parameter of interest. The key property of the parametrization is that it generates a Neyman orthogonal moment condition meaning that the low-dimensional parameter is less sensitive to the estimation of nuisance parameters. Our large sample analysis supports this claim. In particular, we derive sufficient conditions under which the posterior for the low-dimensional parameter contracts around the truth at the parametric rate and is asymptotically normal with a variance that coincides with the semiparametric efficiency bound. These conditions allow for a larger class of nuisance parameters relative to the original parametrization of the regression model. Overall, we conclude that a parametrization that embeds Neyman orthogonality can be a useful device for debiasing posterior distributions in semipa
    

