# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Probabilistic Approach for Alignment with Human Comparisons](https://arxiv.org/abs/2403.10771) | 通过提出的两阶段“监督微调+人类比较”框架，本文研究了如何有效利用人类比较来改善AI模型的对齐，特别是在面对嘈杂数据和高维模型时。 |
| [^2] | [DCRMTA: Unbiased Causal Representation for Multi-touch Attribution.](http://arxiv.org/abs/2401.08875) | DCRMTA提出了一种无偏的多触点归因方法，通过建立转化预测模型和构建对照触点序列来减轻偏差的影响。 |

# 详细

[^1]: 一种基于概率的人类比较对齐方法

    A Probabilistic Approach for Alignment with Human Comparisons

    [https://arxiv.org/abs/2403.10771](https://arxiv.org/abs/2403.10771)

    通过提出的两阶段“监督微调+人类比较”框架，本文研究了如何有效利用人类比较来改善AI模型的对齐，特别是在面对嘈杂数据和高维模型时。

    

    一个增长的趋势是将人类知识整合到学习框架中，利用微妙的人类反馈来完善AI模型。尽管取得了这些进展，但尚未开发出描述人类比较何时改善传统监督微调过程的特定条件的全面理论框架。为弥补这一差距，本文研究了有效利用人类比较来解决由嘈杂数据和高维模型引起的限制。我们提出了一个将机器学习与人类反馈通过概率二分方法联系起来的两阶段“监督微调+人类比较”（SFT+HC）框架。这两阶段框架首先通过SFT过程从带有噪声标记的数据中学习低维表示，然后利用人类比较来改进模型对齐。为了检验对齐阶段的效力，我们引入了一个新概念，称为“标签噪声到一致性”

    arXiv:2403.10771v1 Announce Type: new  Abstract: A growing trend involves integrating human knowledge into learning frameworks, leveraging subtle human feedback to refine AI models. Despite these advances, no comprehensive theoretical framework describing the specific conditions under which human comparisons improve the traditional supervised fine-tuning process has been developed. To bridge this gap, this paper studies the effective use of human comparisons to address limitations arising from noisy data and high-dimensional models. We propose a two-stage "Supervised Fine Tuning+Human Comparison" (SFT+HC) framework connecting machine learning with human feedback through a probabilistic bisection approach. The two-stage framework first learns low-dimensional representations from noisy-labeled data via an SFT procedure, and then uses human comparisons to improve the model alignment. To examine the efficacy of the alignment phase, we introduce a novel concept termed the "label-noise-to-co
    
[^2]: DCRMTA: 无偏的多触点归因的因果表示

    DCRMTA: Unbiased Causal Representation for Multi-touch Attribution. (arXiv:2401.08875v1 [cs.LG])

    [http://arxiv.org/abs/2401.08875](http://arxiv.org/abs/2401.08875)

    DCRMTA提出了一种无偏的多触点归因方法，通过建立转化预测模型和构建对照触点序列来减轻偏差的影响。

    

    多触点归因（MTA）在实现对每个广告触点对于转化行为的贡献的公正估计方面起着关键作用，深刻影响预算分配和广告推荐。传统的多触点归因方法首先构建一个转化预测模型，通过历史数据学习触点序列和用户购买行为之间的内在关系。在此基础上，从原始序列子集中构建对照触点序列，并使用预测模型估计转化，从而计算广告贡献。这些方法的一个隐含假设是转化预测模型的无偏性。然而，由于用户偏好和互联网推荐机制（如过去的购物记录导致的广告推荐同质化）引起的混杂变量因素，转化中很容易产生偏差。

    Multi-touch attribution (MTA) currently plays a pivotal role in achieving a fair estimation of the contributions of each advertising touchpoint to-wards conversion behavior, deeply influencing budget allocation and advertising recommenda-tion. Traditional multi-touch attribution methods initially build a conversion prediction model, an-ticipating learning the inherent relationship be-tween touchpoint sequences and user purchasing behavior through historical data. Based on this, counterfactual touchpoint sequences are con-structed from the original sequence subset, and conversions are estimated using the prediction model, thus calculating advertising contributions. A covert assumption of these methods is the un-biased nature of conversion prediction models. However, due to confounding variables factors arising from user preferences and internet recom-mendation mechanisms such as homogenization of ad recommendations resulting from past shop-ping records, bias can easily occur in conversi
    

