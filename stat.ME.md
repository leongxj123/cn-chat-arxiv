# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards Enhanced Local Explainability of Random Forests: a Proximity-Based Approach.](http://arxiv.org/abs/2310.12428) | 这项研究提出了一种利用随机森林模型的特征空间中的邻近性来解释模型预测的方法，为模型预测提供了局部的解释性，与现有方法相辅相成。通过实验证明了这种方法在债券定价模型中的有效性。 |

# 详细

[^1]: 实现随机森林的局部可解释性增强：基于邻近性的方法

    Towards Enhanced Local Explainability of Random Forests: a Proximity-Based Approach. (arXiv:2310.12428v1 [stat.ML])

    [http://arxiv.org/abs/2310.12428](http://arxiv.org/abs/2310.12428)

    这项研究提出了一种利用随机森林模型的特征空间中的邻近性来解释模型预测的方法，为模型预测提供了局部的解释性，与现有方法相辅相成。通过实验证明了这种方法在债券定价模型中的有效性。

    

    我们提出一种新的方法来解释随机森林（RF）模型的样本外性能，利用了任何RF都可以被表述为自适应加权K最近邻（KNN）模型的事实。具体而言，我们利用RF在特征空间中学到的点之间的邻近性，将随机森林的预测重写为训练数据点目标标签的加权平均值。这种线性性质有助于在训练集观测中为任何模型预测生成属性，从而为RF预测提供了局部的解释性，补充了SHAP等已有方法，这些方法则为特征空间维度上的模型预测生成属性。我们在训练于美国公司债券交易数据的债券定价模型中演示了这种方法，并将其与各种现有的模型解释方法进行了比较。

    We initiate a novel approach to explain the out of sample performance of random forest (RF) models by exploiting the fact that any RF can be formulated as an adaptive weighted K nearest-neighbors model. Specifically, we use the proximity between points in the feature space learned by the RF to re-write random forest predictions exactly as a weighted average of the target labels of training data points. This linearity facilitates a local notion of explainability of RF predictions that generates attributions for any model prediction across observations in the training set, and thereby complements established methods like SHAP, which instead generates attributions for a model prediction across dimensions of the feature space. We demonstrate this approach in the context of a bond pricing model trained on US corporate bond trades, and compare our approach to various existing approaches to model explainability.
    

