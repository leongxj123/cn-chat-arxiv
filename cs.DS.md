# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Smooth Lower Bounds for Differentially Private Algorithms via Padding-and-Permuting Fingerprinting Codes.](http://arxiv.org/abs/2307.07604) | 本论文提出了一种通过填充和置换指纹编码的方法来产生困难实例，从而在各种情景下提供平滑下界。这方法适用于差分隐私平均问题和近似k. |

# 详细

[^1]: 通过填充和置换指纹编码的方法，对差分隐私算法提供平滑下界

    Smooth Lower Bounds for Differentially Private Algorithms via Padding-and-Permuting Fingerprinting Codes. (arXiv:2307.07604v1 [cs.CR])

    [http://arxiv.org/abs/2307.07604](http://arxiv.org/abs/2307.07604)

    本论文提出了一种通过填充和置换指纹编码的方法来产生困难实例，从而在各种情景下提供平滑下界。这方法适用于差分隐私平均问题和近似k.

    

    指纹编码方法是最广泛用于确定约束差分隐私算法的样本复杂度或错误率的方法。然而，对于许多差分隐私问题，我们并不知道适当的下界，并且即使对于我们知道的问题，下界也不平滑，并且通常在误差大于某个阈值时变得无意义。在这项工作中，我们通过将填充和置换转换应用于指纹编码，提出了一种生成困难实例的简单方法。我们通过在不同情景下提供新的下界来说明这种方法的适用性：1. 低准确度情景下差分隐私平均问题的紧密下界，这尤其意味着新的私有1簇问题的下界 2. 近似k

    Fingerprinting arguments, first introduced by Bun, Ullman, and Vadhan (STOC 2014), are the most widely used method for establishing lower bounds on the sample complexity or error of approximately differentially private (DP) algorithms. Still, there are many problems in differential privacy for which we don't know suitable lower bounds, and even for problems that we do, the lower bounds are not smooth, and usually become vacuous when the error is larger than some threshold.  In this work, we present a simple method to generate hard instances by applying a padding-and-permuting transformation to a fingerprinting code. We illustrate the applicability of this method by providing new lower bounds in various settings:  1. A tight lower bound for DP averaging in the low-accuracy regime, which in particular implies a new lower bound for the private 1-cluster problem introduced by Nissim, Stemmer, and Vadhan (PODS 2016).  2. A lower bound on the additive error of DP algorithms for approximate k
    

