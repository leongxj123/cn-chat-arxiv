# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal testing in a class of nonregular models](https://arxiv.org/abs/2403.16413) | 本文研究了一类非常规模型中的最优假设检验，提出了基于似然比过程的渐近一致最强测试方法，通过随机化、调节常数和用户指定备择假设值等方式实现渐近最优性。 |
| [^2] | [Generalization of LiNGAM that allows confounding.](http://arxiv.org/abs/2401.16661) | 本文提出了一种名为LiNGAM-MMI的方法，可以增强LiNGAM模型以处理混淆问题。该方法使用KL散度量化混淆程度，并通过最短路径问题解决方案高效地确定变量顺序，不论是否存在混淆情况。实验证明，LiNGAM-MMI可以更准确地识别正确的变量顺序。 |

# 详细

[^1]: 一类非常规模型中的最优检验

    Optimal testing in a class of nonregular models

    [https://arxiv.org/abs/2403.16413](https://arxiv.org/abs/2403.16413)

    本文研究了一类非常规模型中的最优假设检验，提出了基于似然比过程的渐近一致最强测试方法，通过随机化、调节常数和用户指定备择假设值等方式实现渐近最优性。

    

    本文研究了参数依赖支持的非常规统计模型的最优假设检验。我们考虑了单侧和双侧假设检验，并基于似然比过程发展了渐近一致最强的检验。所提出的单侧检验涉及随机化以实现渐近尺寸控制，一些调节常数以避免在极限似然比过程中的不连续性，并一个用户指定的备择假设值以达到渐近最优性。我们的双侧检验在不施加进一步的限制（如无偏性）的情况下变为渐近一致最强。模拟结果展示了所提出检验的理想功效性质。

    arXiv:2403.16413v1 Announce Type: cross  Abstract: This paper studies optimal hypothesis testing for nonregular statistical models with parameter-dependent support. We consider both one-sided and two-sided hypothesis testing and develop asymptotically uniformly most powerful tests based on the likelihood ratio process. The proposed one-sided test involves randomization to achieve asymptotic size control, some tuning constant to avoid discontinuities in the limiting likelihood ratio process, and a user-specified alternative hypothetical value to achieve the asymptotic optimality. Our two-sided test becomes asymptotically uniformly most powerful without imposing further restrictions such as unbiasedness. Simulation results illustrate desirable power properties of the proposed tests.
    
[^2]: 允许混淆的LiNGAM的泛化

    Generalization of LiNGAM that allows confounding. (arXiv:2401.16661v1 [cs.LG])

    [http://arxiv.org/abs/2401.16661](http://arxiv.org/abs/2401.16661)

    本文提出了一种名为LiNGAM-MMI的方法，可以增强LiNGAM模型以处理混淆问题。该方法使用KL散度量化混淆程度，并通过最短路径问题解决方案高效地确定变量顺序，不论是否存在混淆情况。实验证明，LiNGAM-MMI可以更准确地识别正确的变量顺序。

    

    LiNGAM使用加性噪声模型来确定因果关系的变量顺序，但在混淆方面面临挑战。先前的方法在保持LiNGAM的基本结构的同时，试图识别和处理受混淆影响的变量。结果是，不论是否存在混淆，这些方法都需要大量的计算资源，并且不能确保检测到所有的混淆类型。相比之下，本文通过引入LiNGAM-MMI对LiNGAM进行了增强，该方法使用KL散度量化混淆程度，并安排变量以最小化其影响。该方法通过最短路径问题的形式高效地实现全局最优的变量顺序。在无混淆的情况下，LiNGAM-MMI的处理数据效率与传统LiNGAM相当，同时有效处理混淆情况。我们的实验结果表明，LiNGAM-MMI更准确地确定了正确的变量顺序...

    LiNGAM determines the variable order from cause to effect using additive noise models, but it faces challenges with confounding. Previous methods maintained LiNGAM's fundamental structure while trying to identify and address variables affected by confounding. As a result, these methods required significant computational resources regardless of the presence of confounding, and they did not ensure the detection of all confounding types. In contrast, this paper enhances LiNGAM by introducing LiNGAM-MMI, a method that quantifies the magnitude of confounding using KL divergence and arranges the variables to minimize its impact. This method efficiently achieves a globally optimal variable order through the shortest path problem formulation. LiNGAM-MMI processes data as efficiently as traditional LiNGAM in scenarios without confounding while effectively addressing confounding situations. Our experimental results suggest that LiNGAM-MMI more accurately determines the correct variable order, bo
    

