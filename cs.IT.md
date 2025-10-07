# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generalization of LiNGAM that allows confounding.](http://arxiv.org/abs/2401.16661) | 本文提出了一种名为LiNGAM-MMI的方法，可以增强LiNGAM模型以处理混淆问题。该方法使用KL散度量化混淆程度，并通过最短路径问题解决方案高效地确定变量顺序，不论是否存在混淆情况。实验证明，LiNGAM-MMI可以更准确地识别正确的变量顺序。 |

# 详细

[^1]: 允许混淆的LiNGAM的泛化

    Generalization of LiNGAM that allows confounding. (arXiv:2401.16661v1 [cs.LG])

    [http://arxiv.org/abs/2401.16661](http://arxiv.org/abs/2401.16661)

    本文提出了一种名为LiNGAM-MMI的方法，可以增强LiNGAM模型以处理混淆问题。该方法使用KL散度量化混淆程度，并通过最短路径问题解决方案高效地确定变量顺序，不论是否存在混淆情况。实验证明，LiNGAM-MMI可以更准确地识别正确的变量顺序。

    

    LiNGAM使用加性噪声模型来确定因果关系的变量顺序，但在混淆方面面临挑战。先前的方法在保持LiNGAM的基本结构的同时，试图识别和处理受混淆影响的变量。结果是，不论是否存在混淆，这些方法都需要大量的计算资源，并且不能确保检测到所有的混淆类型。相比之下，本文通过引入LiNGAM-MMI对LiNGAM进行了增强，该方法使用KL散度量化混淆程度，并安排变量以最小化其影响。该方法通过最短路径问题的形式高效地实现全局最优的变量顺序。在无混淆的情况下，LiNGAM-MMI的处理数据效率与传统LiNGAM相当，同时有效处理混淆情况。我们的实验结果表明，LiNGAM-MMI更准确地确定了正确的变量顺序...

    LiNGAM determines the variable order from cause to effect using additive noise models, but it faces challenges with confounding. Previous methods maintained LiNGAM's fundamental structure while trying to identify and address variables affected by confounding. As a result, these methods required significant computational resources regardless of the presence of confounding, and they did not ensure the detection of all confounding types. In contrast, this paper enhances LiNGAM by introducing LiNGAM-MMI, a method that quantifies the magnitude of confounding using KL divergence and arranges the variables to minimize its impact. This method efficiently achieves a globally optimal variable order through the shortest path problem formulation. LiNGAM-MMI processes data as efficiently as traditional LiNGAM in scenarios without confounding while effectively addressing confounding situations. Our experimental results suggest that LiNGAM-MMI more accurately determines the correct variable order, bo
    

