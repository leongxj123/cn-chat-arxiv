# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Interpretable classifiers for tabular data via discretization and feature selection](https://arxiv.org/abs/2402.05680) | 通过离散化和特征选择的方法，我们提出了一种从表格数据中计算出准确又易解释的分类器的方法。在实验证明该方法在准确度上与随机森林和XGBoost等现有方法相当，并且在多种情况下实际上超过了参考结果。 |

# 详细

[^1]: 通过离散化和特征选择的可解释性表格数据分类器

    Interpretable classifiers for tabular data via discretization and feature selection

    [https://arxiv.org/abs/2402.05680](https://arxiv.org/abs/2402.05680)

    通过离散化和特征选择的方法，我们提出了一种从表格数据中计算出准确又易解释的分类器的方法。在实验证明该方法在准确度上与随机森林和XGBoost等现有方法相当，并且在多种情况下实际上超过了参考结果。

    

    我们引入了一种从表格数据中计算出具有解释性且准确的分类器的方法。所得到的分类器是简短的DNF公式，通过将原始数据离散化为布尔形式，然后使用特征选择结合非常快速的算法来产生最佳的布尔分类器。我们通过14个实验来演示该方法，得到的结果的准确度主要与随机森林、XGBoost以及文献中相同数据集的现有结果相似。在多种情况下，我们的方法实际上在准确度方面优于参考结果，尽管我们研究的主要目标是我们的分类器的即时可解释性。我们还证明了一个关于从现实数据中获得的分类器与来自数据背景分布的最佳分类器相对应的概率的新结果。

    We introduce a method for computing immediately human interpretable yet accurate classifiers from tabular data. The classifiers obtained are short DNF-formulas, computed via first discretizing the original data to Boolean form and then using feature selection coupled with a very fast algorithm for producing the best possible Boolean classifier for the setting. We demonstrate the approach via 14 experiments, obtaining results with accuracies mainly similar to ones obtained via random forests, XGBoost, and existing results for the same datasets in the literature. In several cases, our approach in fact outperforms the reference results in relation to accuracy, even though the main objective of our study is the immediate interpretability of our classifiers. We also prove a new result on the probability that the classifier we obtain from real-life data corresponds to the ideally best classifier with respect to the background distribution the data comes from.
    

