# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The VOROS: Lifting ROC curves to 3D](https://arxiv.org/abs/2402.18689) | 引入第三维度，将ROC曲线提升为ROC曲面，提出VOROS作为2D ROC曲线下面积的3D泛化，可以更好地捕捉不同分类器的成本。 |
| [^2] | [Flexible and efficient spatial extremes emulation via variational autoencoders.](http://arxiv.org/abs/2307.08079) | 本文提出了一种新的空间极端值模型，通过集成在变分自动编码器的结构中，可以灵活、高效地模拟具有非平稳相关性的极端事件。实验证明，在时间效率和性能上，相对于传统的贝叶斯推断和许多具有平稳相关性的空间极端值模型，我们的方法具有优势。 |

# 详细

[^1]: VOROS：将ROC曲线提升到3D

    The VOROS: Lifting ROC curves to 3D

    [https://arxiv.org/abs/2402.18689](https://arxiv.org/abs/2402.18689)

    引入第三维度，将ROC曲线提升为ROC曲面，提出VOROS作为2D ROC曲线下面积的3D泛化，可以更好地捕捉不同分类器的成本。

    

    ROC曲线下面积是一个常用的度量，通常用于排列不同二元分类器的相对性能。然而，正如以前所指出的，当真实类值或误分类成本在两个类别之间高度不平衡时，它可能无法准确捕捉不同分类器的效益。我们引入第三维来捕获这些成本，并以一种自然的方式将ROC曲线提升为ROC曲面。我们研究了这个曲面，并引入了VOROS，即ROC曲面上方的体积，作为2D ROC曲线下面积的3D泛化。对于存在预期成本或类别不平衡边界的问题，我们限制考虑适当子区域的ROC曲面的体积。我们展示了VOROS如何在经典和现代示例数据集上更好地捕捉不同分类器的成本。

    arXiv:2402.18689v1 Announce Type: new  Abstract: The area under the ROC curve is a common measure that is often used to rank the relative performance of different binary classifiers. However, as has been also previously noted, it can be a measure that ill-captures the benefits of different classifiers when either the true class values or misclassification costs are highly unbalanced between the two classes. We introduce a third dimension to capture these costs, and lift the ROC curve to a ROC surface in a natural way. We study both this surface and introduce the VOROS, the volume over this ROC surface, as a 3D generalization of the 2D area under the ROC curve. For problems where there are only bounds on the expected costs or class imbalances, we restrict consideration to the volume of the appropriate subregion of the ROC surface. We show how the VOROS can better capture the costs of different classifiers on both a classical and a modern example dataset.
    
[^2]: 通过变分自动 编码器实现灵活高效的空间极端值模拟

    Flexible and efficient spatial extremes emulation via variational autoencoders. (arXiv:2307.08079v1 [stat.ML])

    [http://arxiv.org/abs/2307.08079](http://arxiv.org/abs/2307.08079)

    本文提出了一种新的空间极端值模型，通过集成在变分自动编码器的结构中，可以灵活、高效地模拟具有非平稳相关性的极端事件。实验证明，在时间效率和性能上，相对于传统的贝叶斯推断和许多具有平稳相关性的空间极端值模型，我们的方法具有优势。

    

    许多现实世界的过程具有复杂的尾依赖结构，这种结构无法使用传统的高斯过程来描述。更灵活的空间极端值模型， 如高斯尺度混合模型和单站点调节模型，具有吸引人的极端依赖性质，但往往难以拟合和模拟。本文中，我们提出了一种新的空间极端值模型，具有灵活和非平稳的相关性属性，并将其集成到变分自动编码器 (extVAE) 的编码-解码结构中。 extVAE 可以作为一个时空模拟器，对潜在的机制模型输出状态的分布进行建模，并产生具有与输入相同属性的输出，尤其是在尾部区域。通过广泛的模拟研究，我们证明我们的extVAE比传统的贝叶斯推断更高效，并且在具有 平稳相关性结构的许多空间极端值模型中表现 更好。

    Many real-world processes have complex tail dependence structures that cannot be characterized using classical Gaussian processes. More flexible spatial extremes models such as Gaussian scale mixtures and single-station conditioning models exhibit appealing extremal dependence properties but are often exceedingly prohibitive to fit and simulate from. In this paper, we develop a new spatial extremes model that has flexible and non-stationary dependence properties, and we integrate it in the encoding-decoding structure of a variational autoencoder (extVAE). The extVAE can be used as a spatio-temporal emulator that characterizes the distribution of potential mechanistic model output states and produces outputs that have the same properties as the inputs, especially in the tail. Through extensive simulation studies, we show that our extVAE is vastly more time-efficient than traditional Bayesian inference while also outperforming many spatial extremes models with a stationary dependence str
    

