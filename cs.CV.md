# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [How many views does your deep neural network use for prediction?](https://rss.arxiv.org/abs/2402.01095) | 本文提出了最小有效视图（MSVs）的概念，该概念类似于多视图，但适用于实际图像，并且通过实证研究表明，MSV的数量与模型的预测准确性之间存在关系。 |
| [^2] | [NACHOS: Neural Architecture Search for Hardware Constrained Early Exit Neural Networks.](http://arxiv.org/abs/2401.13330) | NACHOS 是一种面向硬件受限的早期退出神经网络的神经架构搜索方法，可以自动化设计早期退出神经网络并考虑骨干和早期退出分类器之间的关系。 |

# 详细

[^1]: 深度神经网络预测使用多少个视图？

    How many views does your deep neural network use for prediction?

    [https://rss.arxiv.org/abs/2402.01095](https://rss.arxiv.org/abs/2402.01095)

    本文提出了最小有效视图（MSVs）的概念，该概念类似于多视图，但适用于实际图像，并且通过实证研究表明，MSV的数量与模型的预测准确性之间存在关系。

    

    尽管进行了许多理论和实证分析，但深度神经网络（DNN）的泛化能力仍未完全理解。最近，Allen-Zhu和Li（2023）引入了多视图的概念来解释DNN的泛化能力，但他们的主要目标是集成或蒸馏模型，并未讨论用于特定输入预测的多视图估计方法。在本文中，我们提出了最小有效视图（MSVs），它类似于多视图，但可以高效地计算真实图像。MSVs是输入中的一组最小且不同的特征，每个特征保留了模型对该输入的预测。我们通过实证研究表明，不同模型（包括卷积和转换模型）的MSV数量与预测准确性之间存在明确的关系，这表明多视图的角度对于理解（非集成或非蒸馏）DNN的泛化能力也很重要。

    The generalization ability of Deep Neural Networks (DNNs) is still not fully understood, despite numerous theoretical and empirical analyses. Recently, Allen-Zhu & Li (2023) introduced the concept of multi-views to explain the generalization ability of DNNs, but their main target is ensemble or distilled models, and no method for estimating multi-views used in a prediction of a specific input is discussed. In this paper, we propose Minimal Sufficient Views (MSVs), which is similar to multi-views but can be efficiently computed for real images. MSVs is a set of minimal and distinct features in an input, each of which preserves a model's prediction for the input. We empirically show that there is a clear relationship between the number of MSVs and prediction accuracy across models, including convolutional and transformer models, suggesting that a multi-view like perspective is also important for understanding the generalization ability of (non-ensemble or non-distilled) DNNs.
    
[^2]: NACHOS: 硬件受限的早期退出神经网络的神经架构搜索

    NACHOS: Neural Architecture Search for Hardware Constrained Early Exit Neural Networks. (arXiv:2401.13330v1 [cs.LG])

    [http://arxiv.org/abs/2401.13330](http://arxiv.org/abs/2401.13330)

    NACHOS 是一种面向硬件受限的早期退出神经网络的神经架构搜索方法，可以自动化设计早期退出神经网络并考虑骨干和早期退出分类器之间的关系。

    

    早期退出神经网络（EENNs）为标准的深度神经网络（DNN）配备早期退出分类器（EECs），在处理的中间点上提供足够的分类置信度时进行预测。这在效果和效率方面带来了许多好处。目前，EENNs的设计是由专家手动完成的，这是一项复杂和耗时的任务，需要考虑许多方面，包括正确的放置、阈值设置和EECs的计算开销。因此，研究正在探索使用神经架构搜索（NAS）自动化设计EENNs。目前，文献中提出了几个完整的NAS解决方案用于EENNs，并且一个完全自动化的综合设计策略，同时考虑骨干和EECs仍然是一个未解决的问题。为此，本研究呈现了面向硬件受限的早期退出神经网络的神经架构搜索（NACHOS）。

    Early Exit Neural Networks (EENNs) endow astandard Deep Neural Network (DNN) with Early Exit Classifiers (EECs), to provide predictions at intermediate points of the processing when enough confidence in classification is achieved. This leads to many benefits in terms of effectiveness and efficiency. Currently, the design of EENNs is carried out manually by experts, a complex and time-consuming task that requires accounting for many aspects, including the correct placement, the thresholding, and the computational overhead of the EECs. For this reason, the research is exploring the use of Neural Architecture Search (NAS) to automatize the design of EENNs. Currently, few comprehensive NAS solutions for EENNs have been proposed in the literature, and a fully automated, joint design strategy taking into consideration both the backbone and the EECs remains an open problem. To this end, this work presents Neural Architecture Search for Hardware Constrained Early Exit Neural Networks (NACHOS),
    

