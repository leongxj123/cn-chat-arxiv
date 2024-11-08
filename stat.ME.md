# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Differentially Private Range Queries with Correlated Input Perturbation](https://arxiv.org/abs/2402.07066) | 本研究提出了一种具有相关输入扰动的差分隐私范围查询的局部机制，通过级联采样算法实现，实验表明在保障近乎最优的效用的同时，与输出扰动方法在实践中具有竞争力。 |
| [^2] | [Anytime-valid t-tests and confidence sequences for Gaussian means with unknown variance.](http://arxiv.org/abs/2310.03722) | 本文提出了两种新的“e-process”和置信序列方法，分别通过替换Lai的混合方法，并分析了所得结果的宽度。 |

# 详细

[^1]: 具有相关输入扰动的差分隐私范围查询

    Differentially Private Range Queries with Correlated Input Perturbation

    [https://arxiv.org/abs/2402.07066](https://arxiv.org/abs/2402.07066)

    本研究提出了一种具有相关输入扰动的差分隐私范围查询的局部机制，通过级联采样算法实现，实验表明在保障近乎最优的效用的同时，与输出扰动方法在实践中具有竞争力。

    

    本工作提出了一种用于线性查询的局部差分隐私机制，特别是范围查询，利用相关输入扰动同时实现无偏性、一致性、统计透明性和对精度目标的控制，无论是在某些查询边缘上还是在层次数据库结构所暗示的精度要求上。所提出的级联采样算法准确高效地实现了该机制。我们的界限表明，我们在保障近乎最优的效用的同时，与输出扰动方法在实践中具有竞争力。

    This work proposes a class of locally differentially private mechanisms for linear queries, in particular range queries, that leverages correlated input perturbation to simultaneously achieve unbiasedness, consistency, statistical transparency, and control over utility requirements in terms of accuracy targets expressed either in certain query margins or as implied by the hierarchical database structure. The proposed Cascade Sampling algorithm instantiates the mechanism exactly and efficiently. Our bounds show that we obtain near-optimal utility while being empirically competitive against output perturbation methods.
    
[^2]: 未知方差下的高斯均值的任意有效T检验和置信序列

    Anytime-valid t-tests and confidence sequences for Gaussian means with unknown variance. (arXiv:2310.03722v1 [math.ST])

    [http://arxiv.org/abs/2310.03722](http://arxiv.org/abs/2310.03722)

    本文提出了两种新的“e-process”和置信序列方法，分别通过替换Lai的混合方法，并分析了所得结果的宽度。

    

    在1976年，Lai构造了一个非平凡的均值$\mu$的高斯分布的置信序列，该分布的方差$\sigma$是未知的。他使用了关于$\sigma$的不适当（右Haar）混合和关于$\mu$的不适当（平坦）混合。在本文中，我们详细说明了他构建的细节，其中使用了广义的不可积分鞅和扩展的维尔不等式。尽管这确实产生了一个顺序T检验，但由于他的鞅不可积分，它并没有产生一个“e-process”。在本文中，我们为相同的设置开发了两个新的“e-process”和置信序列：一个是在缩减滤波器中的测试鞅，另一个是在规范数据滤波器中的“e-process”。这些分别是通过将Lai的平坦混合替换为高斯混合，并将对$\sigma$的右Haar混合替换为在零空间下的最大似然估计，就像在通用推断中一样。我们还分析了所得结果的宽度。

    In 1976, Lai constructed a nontrivial confidence sequence for the mean $\mu$ of a Gaussian distribution with unknown variance $\sigma$. Curiously, he employed both an improper (right Haar) mixture over $\sigma$ and an improper (flat) mixture over $\mu$. Here, we elaborate carefully on the details of his construction, which use generalized nonintegrable martingales and an extended Ville's inequality. While this does yield a sequential t-test, it does not yield an ``e-process'' (due to the nonintegrability of his martingale). In this paper, we develop two new e-processes and confidence sequences for the same setting: one is a test martingale in a reduced filtration, while the other is an e-process in the canonical data filtration. These are respectively obtained by swapping Lai's flat mixture for a Gaussian mixture, and swapping the right Haar mixture over $\sigma$ with the maximum likelihood estimate under the null, as done in universal inference. We also analyze the width of resulting 
    

