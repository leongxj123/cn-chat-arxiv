# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FreDF: Learning to Forecast in Frequency Domain](https://arxiv.org/abs/2402.02399) | FreDF是一种在频域中学习预测的方法，解决了时间序列建模中标签序列的自相关问题，相比现有方法有更好的性能表现，并且与各种预测模型兼容。 |
| [^2] | [Probabilistic Truly Unordered Rule Sets.](http://arxiv.org/abs/2401.09918) | 本论文提出了概率性真正无序规则集（TURS）方法，用于解决规则集学习中的三个缺点：强加顺序、重叠冲突和多类别目标分类问题。通过利用规则集的概率特性来解决重叠冲突，并形式化定义学习问题。 |

# 详细

[^1]: FreDF: 在频域中学习预测

    FreDF: Learning to Forecast in Frequency Domain

    [https://arxiv.org/abs/2402.02399](https://arxiv.org/abs/2402.02399)

    FreDF是一种在频域中学习预测的方法，解决了时间序列建模中标签序列的自相关问题，相比现有方法有更好的性能表现，并且与各种预测模型兼容。

    

    时间序列建模在历史序列和标签序列中都面临自相关的挑战。当前的研究主要集中在处理历史序列中的自相关问题，但往往忽视了标签序列中的自相关存在。具体来说，新兴的预测模型主要遵循直接预测（DF）范式，在标签序列中假设条件独立性下生成多步预测。这种假设忽视了标签序列中固有的自相关性，从而限制了基于DF的模型的性能。针对这一问题，我们引入了频域增强直接预测（FreDF），通过在频域中学习预测来避免标签自相关的复杂性。我们的实验证明，FreDF在性能上大大超过了包括iTransformer在内的现有最先进方法，并且与各种预测模型兼容。

    Time series modeling is uniquely challenged by the presence of autocorrelation in both historical and label sequences. Current research predominantly focuses on handling autocorrelation within the historical sequence but often neglects its presence in the label sequence. Specifically, emerging forecast models mainly conform to the direct forecast (DF) paradigm, generating multi-step forecasts under the assumption of conditional independence within the label sequence. This assumption disregards the inherent autocorrelation in the label sequence, thereby limiting the performance of DF-based models. In response to this gap, we introduce the Frequency-enhanced Direct Forecast (FreDF), which bypasses the complexity of label autocorrelation by learning to forecast in the frequency domain. Our experiments demonstrate that FreDF substantially outperforms existing state-of-the-art methods including iTransformer and is compatible with a variety of forecast models.
    
[^2]: 概率性真正无序规则集

    Probabilistic Truly Unordered Rule Sets. (arXiv:2401.09918v1 [cs.LG])

    [http://arxiv.org/abs/2401.09918](http://arxiv.org/abs/2401.09918)

    本论文提出了概率性真正无序规则集（TURS）方法，用于解决规则集学习中的三个缺点：强加顺序、重叠冲突和多类别目标分类问题。通过利用规则集的概率特性来解决重叠冲突，并形式化定义学习问题。

    

    最近人们经常重视规则集学习，因为它具有可解释性。然而，现有的方法存在一些缺点。首先，大多数现有方法在规则之间明确或隐含地强加顺序，这使得模型更难以理解。其次，由于处理重叠引起的冲突（即，被多个规则覆盖的实例）的困难，现有方法通常不考虑概率规则。第三，对于多类别目标的学习分类规则研究不足，因为大多数现有方法专注于二分类或通过"一对其余"方法进行多类别分类。为了解决这些缺点，我们提出了TURS，即真正无序规则集。为了解决重叠规则引起的冲突，我们提出了一种新颖的模型，利用我们的规则集的概率特性，只有当它们具有相似的概率输出时允许规则重叠。我们接下来对学习问题进行了形式化定义。

    Rule set learning has recently been frequently revisited because of its interpretability. Existing methods have several shortcomings though. First, most existing methods impose orders among rules, either explicitly or implicitly, which makes the models less comprehensible. Second, due to the difficulty of handling conflicts caused by overlaps (i.e., instances covered by multiple rules), existing methods often do not consider probabilistic rules. Third, learning classification rules for multi-class target is understudied, as most existing methods focus on binary classification or multi-class classification via the ``one-versus-rest" approach.  To address these shortcomings, we propose TURS, for Truly Unordered Rule Sets. To resolve conflicts caused by overlapping rules, we propose a novel model that exploits the probabilistic properties of our rule sets, with the intuition of only allowing rules to overlap if they have similar probabilistic outputs. We next formalize the problem of lear
    

