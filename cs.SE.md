# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Survey of Neural Code Intelligence: Paradigms, Advances and Beyond](https://arxiv.org/abs/2403.14734) | 神经代码智能领域的调查系统回顾了50多种代表性模型和超过680项相关作品，突出了不同研究阶段的范式和技术转变。 |
| [^2] | [Black-Box Prediction of Flaky Test Fix Categories Using Language Models.](http://arxiv.org/abs/2307.00012) | 本文提出了一个使用语言模型的框架，可以自动生成易出错测试的标记数据集，并通过分析测试代码来预测测试的修复类别。实验结果表明UniXcoder优于CodeBERT。 |
| [^3] | [Analysis of Failures and Risks in Deep Learning Model Converters: A Case Study in the ONNX Ecosystem.](http://arxiv.org/abs/2303.17708) | 本文详细分析了深度学习模型转换器的故障情况，特别是对ONNX相关的转换器进行了首次故障分析，并详细报告了故障的症状，原因和位置以及随时间的趋势。 |

# 详细

[^1]: 一项神经代码智能的调查：范式、进展与未来

    A Survey of Neural Code Intelligence: Paradigms, Advances and Beyond

    [https://arxiv.org/abs/2403.14734](https://arxiv.org/abs/2403.14734)

    神经代码智能领域的调查系统回顾了50多种代表性模型和超过680项相关作品，突出了不同研究阶段的范式和技术转变。

    

    arXiv:2403.14734v1 公告类型: 跨领域 摘要: 神经代码智能--利用深度学习理解、生成和优化代码--在整个社会上具有巨大的潜力，可产生深远影响。作为自然语言和编程语言之间的桥梁，这一领域在过去几年引起了两个研究社区研究人员的极大关注。本调查系统地和按时间顺序回顾了代码智能方面的进展，包括50多种代表性模型及其变体、20多种任务类别以及超过680项相关作品。我们遵循历史进展，跟踪不同研究阶段的范式转变（例如，从使用循环神经网络对代码建模到大型语言模型时代）。同时，我们重点介绍了不同阶段涵盖的模型、任务和评估的主要技术转变。对于应用，我们

    arXiv:2403.14734v1 Announce Type: cross  Abstract: Neural Code Intelligence -- leveraging deep learning to understand, generate, and optimize code -- holds immense potential for transformative impacts on the whole society. Bridging the gap between Natural Language and Programming Language, this domain has drawn significant attention from researchers in both research communities over the past few years. This survey presents a systematic and chronological review of the advancements in code intelligence, encompassing over 50 representative models and their variants, more than 20 categories of tasks, and an extensive coverage of over 680 related works. We follow the historical progression to trace the paradigm shifts across different research phases (e.g., from modeling code with recurrent neural networks to the era of Large Language Models). Concurrently, we highlight the major technical transitions in models, tasks, and evaluations spanning through different stages. For applications, we 
    
[^2]: 使用语言模型的黑盒预测易出错测试修复类别

    Black-Box Prediction of Flaky Test Fix Categories Using Language Models. (arXiv:2307.00012v1 [cs.SE])

    [http://arxiv.org/abs/2307.00012](http://arxiv.org/abs/2307.00012)

    本文提出了一个使用语言模型的框架，可以自动生成易出错测试的标记数据集，并通过分析测试代码来预测测试的修复类别。实验结果表明UniXcoder优于CodeBERT。

    

    易出错测试会在相同软件版本的测试下非确定性地通过或失败，引起混乱并浪费开发者时间。尽管机器学习模型已经被用于预测易出错性及其根本原因，但在提供修复支持方面仍有较少工作。为了填补这一空白，我们提出了一个框架，通过仅分析测试代码自动生成13个修复类别的标记数据集，并训练模型来预测易出错测试的修复类别。虽然在当前阶段准确预测修复本身是不现实的，但这些类别提供了关于需要检查的测试代码部分的精确指导。我们的方法基于语言模型，即CodeBERT和UniXcoder，其输出经过前馈神经网络（FNN）或基于孪生网络的Few Shot Learning（FSL）进行了微调。我们的实验结果表明，UniXcoder在正确预测大多数修复类别方面表现优于CodeBERT。

    Flaky tests are problematic because they non-deterministically pass or fail for the same software version under test, causing confusion and wasting developer time. While machine learning models have been used to predict flakiness and its root causes, there is less work on providing support to fix the problem. To address this gap, we propose a framework that automatically generates labeled datasets for 13 fix categories and train models to predict the fix category of a flaky test by analyzing the test code only. Though it is unrealistic at this stage to accurately predict the fix itself, the categories provide precise guidance about what part of the test code to look at. Our approach is based on language models, namely CodeBERT and UniXcoder, whose output is fine-tuned with a Feed Forward Neural Network (FNN) or a Siamese Network-based Few Shot Learning (FSL). Our experimental results show that UniXcoder outperforms CodeBERT, in correctly predicting most of the categories of fixes a dev
    
[^3]: 深度学习模型转换器中的故障和风险分析：以ONNX生态系统的案例研究为例

    Analysis of Failures and Risks in Deep Learning Model Converters: A Case Study in the ONNX Ecosystem. (arXiv:2303.17708v1 [cs.SE])

    [http://arxiv.org/abs/2303.17708](http://arxiv.org/abs/2303.17708)

    本文详细分析了深度学习模型转换器的故障情况，特别是对ONNX相关的转换器进行了首次故障分析，并详细报告了故障的症状，原因和位置以及随时间的趋势。

    

    软件工程师开发，优化和部署深度学习模型。他们在各种开发框架中使用和重新使用模型，并在各种运行时环境中部署它们。在这个多样化的生态系统中，工程师使用深度学习模型转换器将模型从框架移动到运行时环境。然而，转换器中的错误可能会影响模型质量并破坏部署。深度学习模型转换器的故障频率和故障模式尚不清楚。本文针对ONNX (Open Neural Network eXchange)相关的模型转换器进行了首次故障分析。具体而言，我们分析了ONNX转换器在两个重要的DL框架PyTorch和TensorFlow中的过去故障。还报告了故障（N=200个问题）的症状，原因和位置以及随时间的趋势。我们还通过转换8,797个模型（真实世界和人工生成的实例）来评估当今的故障。

    Software engineers develop, fine-tune, and deploy deep learning (DL) models. They use and re-use models in a variety of development frameworks and deploy them on a range of runtime environments. In this diverse ecosystem, engineers use DL model converters to move models from frameworks to runtime environments. However, errors in converters can compromise model quality and disrupt deployment. The failure frequency and failure modes of DL model converters are unknown.  In this paper, we conduct the first failure analysis on DL model converters. Specifically, we characterize failures in model converters associated with ONNX (Open Neural Network eXchange). We analyze past failures in the ONNX converters in two major DL frameworks, PyTorch and TensorFlow. The symptoms, causes, and locations of failures (for N=200 issues), and trends over time are also reported. We also evaluate present-day failures by converting 8,797 models, both real-world and synthetically generated instances. The consis
    

