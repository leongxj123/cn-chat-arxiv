# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [How Much Does Each Datapoint Leak Your Privacy? Quantifying the Per-datum Membership Leakage](https://arxiv.org/abs/2402.10065) | 本论文研究了每个数据点的成员推断攻击，量化了每个数据点的成员泄露，并评估了两种隐私防御措施的效果。 |

# 详细

[^1]: 每个数据点泄露您隐私的程度有多大？量化每个数据点的成员泄露

    How Much Does Each Datapoint Leak Your Privacy? Quantifying the Per-datum Membership Leakage

    [https://arxiv.org/abs/2402.10065](https://arxiv.org/abs/2402.10065)

    本论文研究了每个数据点的成员推断攻击，量化了每个数据点的成员泄露，并评估了两种隐私防御措施的效果。

    

    我们研究了每个数据点的成员推断攻击（MIAs），其中攻击者旨在推断出一个固定目标数据是否已包含在算法的输入数据集中，从而侵犯隐私。首先，我们定义每个数据点的成员泄露为最优对手辨识它的优势。然后，我们量化了经验均值的每个数据点的成员泄露，并表明它取决于目标数据点和数据生成分布之间的马氏距离。我们进一步评估了两种隐私防御措施的效果，即添加高斯噪声和子采样。我们准确地量化了它们都如何降低每个数据点的成员泄露。我们的分析建立在一个结合了似然比检验的Edgeworth展开和Lindeberg-Feller中心极限定理的新型证明技术上。我们的分析连接了现有的似然比和标量乘积攻击，并对这些攻击进行了论证。

    arXiv:2402.10065v1 Announce Type: new  Abstract: We study the per-datum Membership Inference Attacks (MIAs), where an attacker aims to infer whether a fixed target datum has been included in the input dataset of an algorithm and thus, violates privacy. First, we define the membership leakage of a datum as the advantage of the optimal adversary targeting to identify it. Then, we quantify the per-datum membership leakage for the empirical mean, and show that it depends on the Mahalanobis distance between the target datum and the data-generating distribution. We further assess the effect of two privacy defences, i.e. adding Gaussian noise and sub-sampling. We quantify exactly how both of them decrease the per-datum membership leakage. Our analysis builds on a novel proof technique that combines an Edgeworth expansion of the likelihood ratio test and a Lindeberg-Feller central limit theorem. Our analysis connects the existing likelihood ratio and scalar product attacks, and also justifies 
    

