# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Equivalence of the Empirical Risk Minimization to Regularization on the Family of f-Divergences](https://arxiv.org/abs/2402.00501) | 经验风险最小化与f-分布族的正则化的解决方案在特定条件下是唯一的，并且可以通过使用不同的f-分布正则化等效地表示。 |
| [^2] | [The Representation Jensen-Shannon Divergence.](http://arxiv.org/abs/2305.16446) | 本文提出了一种基于表示的新型散度——表示Jensen-Shannon散度，通过将数据分布嵌入到RKHS中，并利用表示的协方差算子的频谱，实现对数据分布的估计，并提供了具有灵活性，可扩展性，可微分性的经验协方差矩阵估计函数和基于核矩阵的估计函数。 |

# 详细

[^1]: 经验风险最小化与f-分布族正则化的等价性

    Equivalence of the Empirical Risk Minimization to Regularization on the Family of f-Divergences

    [https://arxiv.org/abs/2402.00501](https://arxiv.org/abs/2402.00501)

    经验风险最小化与f-分布族的正则化的解决方案在特定条件下是唯一的，并且可以通过使用不同的f-分布正则化等效地表示。

    

    在对f中的温和条件下，给出了经验风险最小化与f-分布的正则化（ERM-$f$DR）的解法。在这些条件下，最优测度被证明是唯一的。并给出了特定选择函数f的解决方案的示例。通过利用f-分布族的灵活性，获得了先前对常见正则化选择的已知解决方案，包括相对熵正则化的唯一解（Type-I和Type-II）。对解的分析揭示了在ERM-$f$DR问题中使用f-分布时的以下属性：$i)$ f-分布正则化强制将解的支持与参考测度的支持重合，引入了在训练数据提供的证据中占主导地位的强归纳偏差；$ii)$ 任何f-分布的正则化都等价于另一种f-分布的正则化。

    The solution to empirical risk minimization with $f$-divergence regularization (ERM-$f$DR) is presented under mild conditions on $f$. Under such conditions, the optimal measure is shown to be unique. Examples of the solution for particular choices of the function $f$ are presented. Previously known solutions to common regularization choices are obtained by leveraging the flexibility of the family of $f$-divergences. These include the unique solutions to empirical risk minimization with relative entropy regularization (Type-I and Type-II). The analysis of the solution unveils the following properties of $f$-divergences when used in the ERM-$f$DR problem: $i\bigl)$ $f$-divergence regularization forces the support of the solution to coincide with the support of the reference measure, which introduces a strong inductive bias that dominates the evidence provided by the training data; and $ii\bigl)$ any $f$-divergence regularization is equivalent to a different $f$-divergence regularization 
    
[^2]: 基于表示的Jensen-Shannon散度

    The Representation Jensen-Shannon Divergence. (arXiv:2305.16446v1 [cs.LG])

    [http://arxiv.org/abs/2305.16446](http://arxiv.org/abs/2305.16446)

    本文提出了一种基于表示的新型散度——表示Jensen-Shannon散度，通过将数据分布嵌入到RKHS中，并利用表示的协方差算子的频谱，实现对数据分布的估计，并提供了具有灵活性，可扩展性，可微分性的经验协方差矩阵估计函数和基于核矩阵的估计函数。

    

    统计散度量化概率分布之间的差异，是机器学习中的一种重要方法。但是，由于数据的底层分布通常未知，从经验样本中估计散度是一个基本难题。本文提出了一种基于再生核希尔伯特空间(RKHS)中协方差算子的新型散度——表示Jensen-Shannon散度。我们的方法将数据分布嵌入到RKHS中，并利用表示的协方差算子的频谱。我们提供了一个从经验协方差矩阵估计的估计函数，它通过使用Fourier特征将数据映射到RKHS中。此估计函数是灵活、可扩展、可微分的，并且适用于小批量优化问题。此外，我们还提供了一种基于核矩阵的估计函数，而不需要对RKHS进行显式映射。我们证明这个量是Jensen-Shannon散度的一个下界。

    Statistical divergences quantify the difference between probability distributions finding multiple uses in machine-learning. However, a fundamental challenge is to estimate divergence from empirical samples since the underlying distributions of the data are usually unknown. In this work, we propose the representation Jensen-Shannon Divergence, a novel divergence based on covariance operators in reproducing kernel Hilbert spaces (RKHS). Our approach embeds the data distributions in an RKHS and exploits the spectrum of the covariance operators of the representations. We provide an estimator from empirical covariance matrices by explicitly mapping the data to an RKHS using Fourier features. This estimator is flexible, scalable, differentiable, and suitable for minibatch-based optimization problems. Additionally, we provide an estimator based on kernel matrices without having an explicit mapping to the RKHS. We show that this quantity is a lower bound on the Jensen-Shannon divergence, and 
    

