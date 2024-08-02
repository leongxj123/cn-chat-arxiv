# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Modeling Latent Selection with Structural Causal Models.](http://arxiv.org/abs/2401.06925) | 本文介绍了一种在结构因果模型中对潜在选择进行建模的方法，并展示了它如何帮助进行因果推理任务，包括处理选择偏差。 |
| [^2] | [Pattern Recovery in Penalized and Thresholded Estimation and its Geometry.](http://arxiv.org/abs/2307.10158) | 我们提出了一种惩罚化和阈值化估计的模式恢复方法，并定义了模式和恢复条件。对于LASSO，无噪声恢复条件和互不表示条件起到了相同的作用。 |

# 详细

[^1]: 用结构因果模型对潜在选择进行建模

    Modeling Latent Selection with Structural Causal Models. (arXiv:2401.06925v1 [cs.AI])

    [http://arxiv.org/abs/2401.06925](http://arxiv.org/abs/2401.06925)

    本文介绍了一种在结构因果模型中对潜在选择进行建模的方法，并展示了它如何帮助进行因果推理任务，包括处理选择偏差。

    

    选择偏倚在现实世界的数据中是普遍存在的，如果不正确处理可能导致误导性结果。我们引入了对结构因果模型（SCMs）进行条件操作的方法，以从因果的角度对潜在选择进行建模。我们展示了条件操作将具有明确潜在选择机制的SCM转换为没有此类选择机制的SCM，这在一定程度上编码了根据原始SCM选择的亚总体的因果语义。此外，我们还展示了该条件操作保持SCMs的简洁性，无环性和线性性，并与边际化操作相符合。由于这些特性与边际化和干预结合起来，条件操作为在潜在细节已经去除的因果模型中进行因果推理任务提供了一个有价值的工具。我们通过例子演示了如何将因果推断的经典结果推广以包括选择偏倚。

    Selection bias is ubiquitous in real-world data, and can lead to misleading results if not dealt with properly. We introduce a conditioning operation on Structural Causal Models (SCMs) to model latent selection from a causal perspective. We show that the conditioning operation transforms an SCM with the presence of an explicit latent selection mechanism into an SCM without such selection mechanism, which partially encodes the causal semantics of the selected subpopulation according to the original SCM. Furthermore, we show that this conditioning operation preserves the simplicity, acyclicity, and linearity of SCMs, and commutes with marginalization. Thanks to these properties, combined with marginalization and intervention, the conditioning operation offers a valuable tool for conducting causal reasoning tasks within causal models where latent details have been abstracted away. We demonstrate by example how classical results of causal inference can be generalized to include selection b
    
[^2]: 惩罚化和阈值化估计中的模式恢复及其几何

    Pattern Recovery in Penalized and Thresholded Estimation and its Geometry. (arXiv:2307.10158v1 [math.ST])

    [http://arxiv.org/abs/2307.10158](http://arxiv.org/abs/2307.10158)

    我们提出了一种惩罚化和阈值化估计的模式恢复方法，并定义了模式和恢复条件。对于LASSO，无噪声恢复条件和互不表示条件起到了相同的作用。

    

    我们考虑惩罚估计的框架，其中惩罚项由实值的多面体规范给出，其中包括诸如LASSO（以及其许多变体如广义LASSO）、SLOPE、OSCAR、PACS等方法。每个估计器可以揭示未知参数向量的不同结构或“模式”。我们定义了基于次微分的模式的一般概念，并形式化了一种衡量其复杂性的方法。对于模式恢复，我们提供了一个特定模式以正概率被该过程检测到的最小条件，即所谓的可达性条件。利用我们的方法，我们还引入了更强的无噪声恢复条件。对于LASSO，众所周知，互不表示条件是使模式恢复的概率大于1/2所必需的，并且我们展示了无噪声恢复起到了完全相同的作用，从而扩展和统一了互不表示条件。

    We consider the framework of penalized estimation where the penalty term is given by a real-valued polyhedral gauge, which encompasses methods such as LASSO (and many variants thereof such as the generalized LASSO), SLOPE, OSCAR, PACS and others. Each of these estimators can uncover a different structure or ``pattern'' of the unknown parameter vector. We define a general notion of patterns based on subdifferentials and formalize an approach to measure their complexity. For pattern recovery, we provide a minimal condition for a particular pattern to be detected by the procedure with positive probability, the so-called accessibility condition. Using our approach, we also introduce the stronger noiseless recovery condition. For the LASSO, it is well known that the irrepresentability condition is necessary for pattern recovery with probability larger than $1/2$ and we show that the noiseless recovery plays exactly the same role, thereby extending and unifying the irrepresentability conditi
    

