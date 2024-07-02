# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unmasking Bias and Inequities: A Systematic Review of Bias Detection and Mitigation in Healthcare Artificial Intelligence Using Electronic Health Records.](http://arxiv.org/abs/2310.19917) | 本综述对涉及利用电子健康记录数据的医疗人工智能研究中的偏见进行了系统综述，共涵盖了六种主要的偏见类型，同时总结了现有的偏见处理方法。 |
| [^2] | [Uncertainty Quantification for Molecular Property Predictions with Graph Neural Architecture Search.](http://arxiv.org/abs/2307.10438) | 用于分子属性预测的图神经网络方法通常无法量化预测的不确定性，本研究提出了一种自动化的不确定性量化方法AutoGNNUQ，通过架构搜索生成高性能的图神经网络集合，并利用方差分解将数据和模型的不确定性分开，从而提供了减少不确定性的有价值见解。 |

# 详细

[^1]: 揭示偏见和不平等：利用电子健康记录的医疗人工智能中偏见检测和缓解的系统综述

    Unmasking Bias and Inequities: A Systematic Review of Bias Detection and Mitigation in Healthcare Artificial Intelligence Using Electronic Health Records. (arXiv:2310.19917v1 [cs.AI])

    [http://arxiv.org/abs/2310.19917](http://arxiv.org/abs/2310.19917)

    本综述对涉及利用电子健康记录数据的医疗人工智能研究中的偏见进行了系统综述，共涵盖了六种主要的偏见类型，同时总结了现有的偏见处理方法。

    

    目的：利用电子健康记录的人工智能应用在医疗领域越来越受到欢迎，但也引入了各种类型的偏见。本研究旨在系统综述涉及利用电子健康记录数据的人工智能研究中的偏见。方法：遵循Preferred Reporting Items for Systematic Reviews and Meta-analyses (PRISMA)准则进行了系统综述。从PubMed、Web of Science和电气和电子工程师学会中检索了2010年1月1日至2022年10月31日期间发表的文章。我们定义了六种主要的偏见类型，并总结了现有的偏见处理方法。结果：在检索到的252篇文章中，有20篇符合最终综述的纳入标准。本综述涵盖了六种偏见中的五种：八项研究分析了选择偏见；六项研究针对隐性偏见；五项研究对混杂偏见进行了研究；四项研究对测量偏见进行了研究；两项研究对算法偏见进行了研究。在偏见处理方法方面，有十项研究进行了探讨。

    Objectives: Artificial intelligence (AI) applications utilizing electronic health records (EHRs) have gained popularity, but they also introduce various types of bias. This study aims to systematically review the literature that address bias in AI research utilizing EHR data. Methods: A systematic review was conducted following the Preferred Reporting Items for Systematic Reviews and Meta-analyses (PRISMA) guideline. We retrieved articles published between January 1, 2010, and October 31, 2022, from PubMed, Web of Science, and the Institute of Electrical and Electronics Engineers. We defined six major types of bias and summarized the existing approaches in bias handling. Results: Out of the 252 retrieved articles, 20 met the inclusion criteria for the final review. Five out of six bias were covered in this review: eight studies analyzed selection bias; six on implicit bias; five on confounding bias; four on measurement bias; two on algorithmic bias. For bias handling approaches, ten st
    
[^2]: 用图神经结构搜索进行分子属性预测的不确定性量化

    Uncertainty Quantification for Molecular Property Predictions with Graph Neural Architecture Search. (arXiv:2307.10438v1 [cs.LG])

    [http://arxiv.org/abs/2307.10438](http://arxiv.org/abs/2307.10438)

    用于分子属性预测的图神经网络方法通常无法量化预测的不确定性，本研究提出了一种自动化的不确定性量化方法AutoGNNUQ，通过架构搜索生成高性能的图神经网络集合，并利用方差分解将数据和模型的不确定性分开，从而提供了减少不确定性的有价值见解。

    

    图神经网络（GNN）已成为分子属性预测中突出的数据驱动方法。然而，典型GNN模型的一个关键限制是无法量化预测中的不确定性。这种能力对于确保在下游任务中可信地使用和部署模型至关重要。为此，我们引入了AutoGNNUQ，一种自动化的分子属性预测不确定性量化方法。AutoGNNUQ利用架构搜索生成一组高性能的GNN集合，能够估计预测的不确定性。我们的方法使用方差分解来分离数据（aleatoric）和模型（epistemic）不确定性，为减少它们提供了有价值的见解。在我们的计算实验中，我们展示了AutoGNNUQ在多个基准数据集上在预测准确性和不确定性测量性能方面超过了现有的不确定性量化方法。此外，我们利用t-SNE可视化来解释不确定性的来源和结构。

    Graph Neural Networks (GNNs) have emerged as a prominent class of data-driven methods for molecular property prediction. However, a key limitation of typical GNN models is their inability to quantify uncertainties in the predictions. This capability is crucial for ensuring the trustworthy use and deployment of models in downstream tasks. To that end, we introduce AutoGNNUQ, an automated uncertainty quantification (UQ) approach for molecular property prediction. AutoGNNUQ leverages architecture search to generate an ensemble of high-performing GNNs, enabling the estimation of predictive uncertainties. Our approach employs variance decomposition to separate data (aleatoric) and model (epistemic) uncertainties, providing valuable insights for reducing them. In our computational experiments, we demonstrate that AutoGNNUQ outperforms existing UQ methods in terms of both prediction accuracy and UQ performance on multiple benchmark datasets. Additionally, we utilize t-SNE visualization to exp
    

