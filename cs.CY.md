# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Ensemble Framework for Explainable Geospatial Machine Learning Models](https://arxiv.org/abs/2403.03328) | 介绍了一个集成框架，结合了本地空间加权方案、可解释人工智能（XAI）和尖端机器学习技术，以提高地理空间机器学习模型的可解释性和准确性。 |
| [^2] | [Properties and Challenges of LLM-Generated Explanations](https://arxiv.org/abs/2402.10532) | 该研究探讨了大型语言模型生成的解释在多领域指导微调数据集上的特性，发现生成的解释表现出选择性和包含说明性元素，但较少是主观或误导性的。 |

# 详细

[^1]: 一个用于可解释地理空间机器学习模型的集成框架

    An Ensemble Framework for Explainable Geospatial Machine Learning Models

    [https://arxiv.org/abs/2403.03328](https://arxiv.org/abs/2403.03328)

    介绍了一个集成框架，结合了本地空间加权方案、可解释人工智能（XAI）和尖端机器学习技术，以提高地理空间机器学习模型的可解释性和准确性。

    

    分析空间变化效应在地理分析中至关重要。然而，由于地理数据的复杂性和非线性，准确捕捉和解释这种变异性是具有挑战性的。在这里，我们介绍了一个集成框架，结合了本地空间加权方案、可解释人工智能（XAI）和尖端机器学习技术，以弥合传统地理分析模型和通用机器学习方法之间的差距。通过对合成数据集的测试，验证了该框架通过阐明空间变异性，提高了地理回归和分类预测的可解释性和准确性。它显著提高了预测精度，提供了一种理解空间现象的新方法。

    arXiv:2403.03328v1 Announce Type: new  Abstract: Analyzing spatial varying effect is pivotal in geographic analysis. Yet, accurately capturing and interpreting this variability is challenging due to the complexity and non-linearity of geospatial data. Herein, we introduce an integrated framework that merges local spatial weighting scheme, Explainable Artificial Intelligence (XAI), and cutting-edge machine learning technologies to bridge the gap between traditional geographic analysis models and general machine learning approaches. Through tests on synthetic datasets, this framework is verified to enhance the interpretability and accuracy of predictions in both geographic regression and classification by elucidating spatial variability. It significantly boosts prediction precision, offering a novel approach to understanding spatial phenomena.
    
[^2]: LLM生成的解释的特性和挑战

    Properties and Challenges of LLM-Generated Explanations

    [https://arxiv.org/abs/2402.10532](https://arxiv.org/abs/2402.10532)

    该研究探讨了大型语言模型生成的解释在多领域指导微调数据集上的特性，发现生成的解释表现出选择性和包含说明性元素，但较少是主观或误导性的。

    

    大型语言模型（LLMs）的自我合理化能力在限定环境中得到了探索，使用特定任务/数据集。然而，当前LLMs并不（仅）依赖于特定注释的数据；然而，它们经常解释它们的输出。生成的解释的特性受预训练语料库和用于指导微调的目标数据的影响。由于预训练语料库包含大量野外人类编写的解释，我们假设LLMs采用了人类解释的共同特性。通过分析多域指导微调数据集的输出，我们发现生成的解释表现出选择性并包含说明性元素，但很少是主观或误导性的。我们讨论了属性存在或缺失的原因和后果。特别是，我们概述了根据LLMs预训练语料库和微调数据的性质，这些属性存在或缺失的积极和消极影响。

    arXiv:2402.10532v1 Announce Type: cross  Abstract: The self-rationalising capabilities of large language models (LLMs) have been explored in restricted settings, using task/specific data sets. However, current LLMs do not (only) rely on specifically annotated data; nonetheless, they frequently explain their outputs. The properties of the generated explanations are influenced by the pre-training corpus and by the target data used for instruction fine-tuning. As the pre-training corpus includes a large amount of human-written explanations "in the wild", we hypothesise that LLMs adopt common properties of human explanations. By analysing the outputs for a multi-domain instruction fine-tuning data set, we find that generated explanations show selectivity and contain illustrative elements, but less frequently are subjective or misleading. We discuss reasons and consequences of the properties' presence or absence. In particular, we outline positive and negative implications depending on the 
    

