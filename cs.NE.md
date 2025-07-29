# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Large-scale Generative AI Models Lack Visual Number Sense](https://arxiv.org/abs/2402.03328) | 本研究调查了基于大规模Transformer架构的生成性AI模型是否能够准确命名物体数量或生成包含目标数量物品的图像，结果发现这些模型都没有以类似人类的方式表现，并且即使对于小数量的物体也会出现显著的错误。 |
| [^2] | [NACHOS: Neural Architecture Search for Hardware Constrained Early Exit Neural Networks.](http://arxiv.org/abs/2401.13330) | NACHOS 是一种面向硬件受限的早期退出神经网络的神经架构搜索方法，可以自动化设计早期退出神经网络并考虑骨干和早期退出分类器之间的关系。 |

# 详细

[^1]: 大规模生成AI模型缺乏视觉数字感知能力

    Large-scale Generative AI Models Lack Visual Number Sense

    [https://arxiv.org/abs/2402.03328](https://arxiv.org/abs/2402.03328)

    本研究调查了基于大规模Transformer架构的生成性AI模型是否能够准确命名物体数量或生成包含目标数量物品的图像，结果发现这些模型都没有以类似人类的方式表现，并且即使对于小数量的物体也会出现显著的错误。

    

    人类能够在视觉场景中轻松判断物体的数量，即使不进行计数，而且这种技能在各种动物物种和语言发展和正式学校教育之前的婴儿中都有记录。对于小的物体集，数字判断是无误的，而对于更大的集合，回应变得近似，并且变异性与目标数字成比例增加。尽管物体特征（如颜色或形状）存在差异，但这种回应模式在所有类型的物体上观察到，这表明我们的视觉数字感知依赖于数字数量的抽象表示。在本研究中，我们调查了基于大规模Transformer架构的生成性人工智能（AI）模型是否可以可靠地命名简单视觉刺激中的物体数量或生成包含目标物品数量的图像（1-10范围内）。令人惊讶的是，所考虑的所有基础模型都没有以类似人类一样的方式表现出来：即使是具有较小数量的物体也会犯下显著的错误。

    Humans can readily judge the number of objects in a visual scene, even without counting, and such a skill has been documented in a variety of animal species and in babies prior to language development and formal schooling. Numerical judgments are error-free for small sets, while for larger collections responses become approximate, with variability increasing proportionally to the target number. This response pattern is observed for items of all kinds, despite variation in object features (such as color or shape), suggesting that our visual number sense relies on abstract representations of numerosity. Here, we investigated whether generative Artificial Intelligence (AI) models based on large-scale transformer architectures can reliably name the number of objects in simple visual stimuli or generate images containing a target number of items in the 1-10 range. Surprisingly, none of the foundation models considered performed in a human-like way: They all made striking errors even with sm
    
[^2]: NACHOS: 硬件受限的早期退出神经网络的神经架构搜索

    NACHOS: Neural Architecture Search for Hardware Constrained Early Exit Neural Networks. (arXiv:2401.13330v1 [cs.LG])

    [http://arxiv.org/abs/2401.13330](http://arxiv.org/abs/2401.13330)

    NACHOS 是一种面向硬件受限的早期退出神经网络的神经架构搜索方法，可以自动化设计早期退出神经网络并考虑骨干和早期退出分类器之间的关系。

    

    早期退出神经网络（EENNs）为标准的深度神经网络（DNN）配备早期退出分类器（EECs），在处理的中间点上提供足够的分类置信度时进行预测。这在效果和效率方面带来了许多好处。目前，EENNs的设计是由专家手动完成的，这是一项复杂和耗时的任务，需要考虑许多方面，包括正确的放置、阈值设置和EECs的计算开销。因此，研究正在探索使用神经架构搜索（NAS）自动化设计EENNs。目前，文献中提出了几个完整的NAS解决方案用于EENNs，并且一个完全自动化的综合设计策略，同时考虑骨干和EECs仍然是一个未解决的问题。为此，本研究呈现了面向硬件受限的早期退出神经网络的神经架构搜索（NACHOS）。

    Early Exit Neural Networks (EENNs) endow astandard Deep Neural Network (DNN) with Early Exit Classifiers (EECs), to provide predictions at intermediate points of the processing when enough confidence in classification is achieved. This leads to many benefits in terms of effectiveness and efficiency. Currently, the design of EENNs is carried out manually by experts, a complex and time-consuming task that requires accounting for many aspects, including the correct placement, the thresholding, and the computational overhead of the EECs. For this reason, the research is exploring the use of Neural Architecture Search (NAS) to automatize the design of EENNs. Currently, few comprehensive NAS solutions for EENNs have been proposed in the literature, and a fully automated, joint design strategy taking into consideration both the backbone and the EECs remains an open problem. To this end, this work presents Neural Architecture Search for Hardware Constrained Early Exit Neural Networks (NACHOS),
    

