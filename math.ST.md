# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The VOROS: Lifting ROC curves to 3D](https://arxiv.org/abs/2402.18689) | 引入第三维度，将ROC曲线提升为ROC曲面，提出VOROS作为2D ROC曲线下面积的3D泛化，可以更好地捕捉不同分类器的成本。 |

# 详细

[^1]: VOROS：将ROC曲线提升到3D

    The VOROS: Lifting ROC curves to 3D

    [https://arxiv.org/abs/2402.18689](https://arxiv.org/abs/2402.18689)

    引入第三维度，将ROC曲线提升为ROC曲面，提出VOROS作为2D ROC曲线下面积的3D泛化，可以更好地捕捉不同分类器的成本。

    

    ROC曲线下面积是一个常用的度量，通常用于排列不同二元分类器的相对性能。然而，正如以前所指出的，当真实类值或误分类成本在两个类别之间高度不平衡时，它可能无法准确捕捉不同分类器的效益。我们引入第三维来捕获这些成本，并以一种自然的方式将ROC曲线提升为ROC曲面。我们研究了这个曲面，并引入了VOROS，即ROC曲面上方的体积，作为2D ROC曲线下面积的3D泛化。对于存在预期成本或类别不平衡边界的问题，我们限制考虑适当子区域的ROC曲面的体积。我们展示了VOROS如何在经典和现代示例数据集上更好地捕捉不同分类器的成本。

    arXiv:2402.18689v1 Announce Type: new  Abstract: The area under the ROC curve is a common measure that is often used to rank the relative performance of different binary classifiers. However, as has been also previously noted, it can be a measure that ill-captures the benefits of different classifiers when either the true class values or misclassification costs are highly unbalanced between the two classes. We introduce a third dimension to capture these costs, and lift the ROC curve to a ROC surface in a natural way. We study both this surface and introduce the VOROS, the volume over this ROC surface, as a 3D generalization of the 2D area under the ROC curve. For problems where there are only bounds on the expected costs or class imbalances, we restrict consideration to the volume of the appropriate subregion of the ROC surface. We show how the VOROS can better capture the costs of different classifiers on both a classical and a modern example dataset.
    

