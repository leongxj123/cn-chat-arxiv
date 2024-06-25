# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Mixture of Mixups for Multi-label Classification of Rare Anuran Sounds](https://arxiv.org/abs/2403.09598) | 该论文提出了利用混合Mixup方法来解决稀有无尾蛙声多标签分类中的挑战，实验证明这种方法在处理类别不平衡和多标签示例方面具有有效性。 |

# 详细

[^1]: 混合Mixup用于稀有无尾蛙声多标签分类

    Mixture of Mixups for Multi-label Classification of Rare Anuran Sounds

    [https://arxiv.org/abs/2403.09598](https://arxiv.org/abs/2403.09598)

    该论文提出了利用混合Mixup方法来解决稀有无尾蛙声多标签分类中的挑战，实验证明这种方法在处理类别不平衡和多标签示例方面具有有效性。

    

    多标签不平衡分类在机器学习中是一个重要挑战，特别在生物声学中尤为明显，动物声音经常同时出现，而某些声音比其他声音要少得多。本文针对使用包含类别不平衡和多标签示例的数据集AnuraSet，专注于分类无尾目物种声音的特定情况。为了解决这些挑战，我们引入了Mixture of Mixups (Mix2)，这是一个利用混合正则化方法Mixup、Manifold Mixup和MultiMix的框架。实验结果表明，这些方法单独使用可能导致次优结果；然而，当随机应用它们时，每次训练迭代选取一个方法，它们在解决提到的挑战方面表现出有效性，特别是对于发生次数较少的稀有类别。进一步分析表明，Mix2在跨各种类别同时出现水平上也能有效分类声音。

    arXiv:2403.09598v1 Announce Type: cross  Abstract: Multi-label imbalanced classification poses a significant challenge in machine learning, particularly evident in bioacoustics where animal sounds often co-occur, and certain sounds are much less frequent than others. This paper focuses on the specific case of classifying anuran species sounds using the dataset AnuraSet, that contains both class imbalance and multi-label examples. To address these challenges, we introduce Mixture of Mixups (Mix2), a framework that leverages mixing regularization methods Mixup, Manifold Mixup, and MultiMix. Experimental results show that these methods, individually, may lead to suboptimal results; however, when applied randomly, with one selected at each training iteration, they prove effective in addressing the mentioned challenges, particularly for rare classes with few occurrences. Further analysis reveals that Mix2 is also proficient in classifying sounds across various levels of class co-occurrences
    

