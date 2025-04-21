# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Confidence on the Focal: Conformal Prediction with Selection-Conditional Coverage](https://arxiv.org/abs/2403.03868) | 该论文提出了一种构建具有有限样本精确覆盖的预测集的通用框架，可以解决在数据驱动情境中由于选择偏差导致的边缘有效预测区间误导问题。 |

# 详细

[^1]: 焦点置信: 带有选择条件覆盖的整体预测

    Confidence on the Focal: Conformal Prediction with Selection-Conditional Coverage

    [https://arxiv.org/abs/2403.03868](https://arxiv.org/abs/2403.03868)

    该论文提出了一种构建具有有限样本精确覆盖的预测集的通用框架，可以解决在数据驱动情境中由于选择偏差导致的边缘有效预测区间误导问题。

    

    整体预测建立在边缘有效的预测区间上，该区间以某种规定的概率覆盖了随机抽取的新测试点的未知结果。在实践中，常见情况是，在看到测试单元后，从业者以数据驱动的方式决定关注哪些测试单元，并希望量化焦点单元的不确定性。在这种情况下，对于这些焦点单元的边缘有效预测区间可能会因选择偏差而具有误导性。本文提出了一个构建具有有限样本精确覆盖的预测集的通用框架，该覆盖是有条件于所选单元的。其一般形式适用于任意选择规则，并将Mondrian整体预测推广到多个测试单元和非等变分类器。然后，我们为多个现实的选择规则计算了适用于我们框架的计算效率实现，包括top-K选择、优化等。

    arXiv:2403.03868v1 Announce Type: cross  Abstract: Conformal prediction builds marginally valid prediction intervals which cover the unknown outcome of a randomly drawn new test point with a prescribed probability. In practice, a common scenario is that, after seeing the test unit(s), practitioners decide which test unit(s) to focus on in a data-driven manner, and wish to quantify the uncertainty for the focal unit(s). In such cases, marginally valid prediction intervals for these focal units can be misleading due to selection bias. This paper presents a general framework for constructing a prediction set with finite-sample exact coverage conditional on the unit being selected. Its general form works for arbitrary selection rules, and generalizes Mondrian Conformal Prediction to multiple test units and non-equivariant classifiers. We then work out computationally efficient implementation of our framework for a number of realistic selection rules, including top-K selection, optimization
    

