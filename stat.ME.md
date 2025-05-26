# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MDI+: A Flexible Random Forest-Based Feature Importance Framework.](http://arxiv.org/abs/2307.01932) | MDI+是一种灵活的基于随机森林的特征重要性框架，通过替换线性回归模型和度量，利用正则化的广义线性模型和更适合数据结构的度量来推广MDI。此外，MDI+还引入了其他特征来减轻决策树对加法或平滑模型的已知偏差。 |

# 详细

[^1]: MDI+:一种灵活的基于随机森林的特征重要性框架

    MDI+: A Flexible Random Forest-Based Feature Importance Framework. (arXiv:2307.01932v1 [stat.ME])

    [http://arxiv.org/abs/2307.01932](http://arxiv.org/abs/2307.01932)

    MDI+是一种灵活的基于随机森林的特征重要性框架，通过替换线性回归模型和度量，利用正则化的广义线性模型和更适合数据结构的度量来推广MDI。此外，MDI+还引入了其他特征来减轻决策树对加法或平滑模型的已知偏差。

    

    以不纯度减少的平均值(MDI)是随机森林(RF)中一种流行的特征重要性评估方法。我们展示了在RF中每个树的特征$X_k$的MDI等价于响应变量在决策树集合上的线性回归的未归一化$R^2$值。我们利用这种解释提出了一种灵活的特征重要性框架MDI+，MDI+通过允许分析人员将线性回归模型和$R^2$度量替换为正则化的广义线性模型(GLM)和更适合给定数据结构的度量来推广MDI。此外，MDI+还引入了其他特征来减轻决策树对加法或平滑模型的已知偏差。我们进一步提供了关于如何基于可预测性、可计算性和稳定性框架选择适当的GLM和度量的指导，以进行真实数据科学研究。大量基于数据的模拟结果显示，MDI+在性能上显著优于传统的MDI。

    Mean decrease in impurity (MDI) is a popular feature importance measure for random forests (RFs). We show that the MDI for a feature $X_k$ in each tree in an RF is equivalent to the unnormalized $R^2$ value in a linear regression of the response on the collection of decision stumps that split on $X_k$. We use this interpretation to propose a flexible feature importance framework called MDI+. Specifically, MDI+ generalizes MDI by allowing the analyst to replace the linear regression model and $R^2$ metric with regularized generalized linear models (GLMs) and metrics better suited for the given data structure. Moreover, MDI+ incorporates additional features to mitigate known biases of decision trees against additive or smooth models. We further provide guidance on how practitioners can choose an appropriate GLM and metric based upon the Predictability, Computability, Stability framework for veridical data science. Extensive data-inspired simulations show that MDI+ significantly outperfor
    

