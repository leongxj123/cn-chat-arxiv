# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Uncertainty Quantification for Molecular Property Predictions with Graph Neural Architecture Search.](http://arxiv.org/abs/2307.10438) | 用于分子属性预测的图神经网络方法通常无法量化预测的不确定性，本研究提出了一种自动化的不确定性量化方法AutoGNNUQ，通过架构搜索生成高性能的图神经网络集合，并利用方差分解将数据和模型的不确定性分开，从而提供了减少不确定性的有价值见解。 |

# 详细

[^1]: 用图神经结构搜索进行分子属性预测的不确定性量化

    Uncertainty Quantification for Molecular Property Predictions with Graph Neural Architecture Search. (arXiv:2307.10438v1 [cs.LG])

    [http://arxiv.org/abs/2307.10438](http://arxiv.org/abs/2307.10438)

    用于分子属性预测的图神经网络方法通常无法量化预测的不确定性，本研究提出了一种自动化的不确定性量化方法AutoGNNUQ，通过架构搜索生成高性能的图神经网络集合，并利用方差分解将数据和模型的不确定性分开，从而提供了减少不确定性的有价值见解。

    

    图神经网络（GNN）已成为分子属性预测中突出的数据驱动方法。然而，典型GNN模型的一个关键限制是无法量化预测中的不确定性。这种能力对于确保在下游任务中可信地使用和部署模型至关重要。为此，我们引入了AutoGNNUQ，一种自动化的分子属性预测不确定性量化方法。AutoGNNUQ利用架构搜索生成一组高性能的GNN集合，能够估计预测的不确定性。我们的方法使用方差分解来分离数据（aleatoric）和模型（epistemic）不确定性，为减少它们提供了有价值的见解。在我们的计算实验中，我们展示了AutoGNNUQ在多个基准数据集上在预测准确性和不确定性测量性能方面超过了现有的不确定性量化方法。此外，我们利用t-SNE可视化来解释不确定性的来源和结构。

    Graph Neural Networks (GNNs) have emerged as a prominent class of data-driven methods for molecular property prediction. However, a key limitation of typical GNN models is their inability to quantify uncertainties in the predictions. This capability is crucial for ensuring the trustworthy use and deployment of models in downstream tasks. To that end, we introduce AutoGNNUQ, an automated uncertainty quantification (UQ) approach for molecular property prediction. AutoGNNUQ leverages architecture search to generate an ensemble of high-performing GNNs, enabling the estimation of predictive uncertainties. Our approach employs variance decomposition to separate data (aleatoric) and model (epistemic) uncertainties, providing valuable insights for reducing them. In our computational experiments, we demonstrate that AutoGNNUQ outperforms existing UQ methods in terms of both prediction accuracy and UQ performance on multiple benchmark datasets. Additionally, we utilize t-SNE visualization to exp
    

