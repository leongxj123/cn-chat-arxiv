# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Transparent and Clinically Interpretable AI for Lung Cancer Detection in Chest X-Rays](https://arxiv.org/abs/2403.19444) | 使用概念瓶颈模型的ante-hoc方法将临床概念引入到分类管道中，提供了肺癌检测决策过程中的有价值见解，相较于基线深度学习模型实现了更好的分类性能（F1 > 0.9）。 |
| [^2] | [Approximate Nullspace Augmented Finetuning for Robust Vision Transformers](https://arxiv.org/abs/2403.10476) | 本研究提出了一种启发自线性代数零空间概念的视觉变换器鲁棒性增强微调方法，通过合成近似零空间元素来提高模型的鲁棒性。 |

# 详细

[^1]: 透明且临床可解释的人工智能用于胸部X射线肺癌检测

    Transparent and Clinically Interpretable AI for Lung Cancer Detection in Chest X-Rays

    [https://arxiv.org/abs/2403.19444](https://arxiv.org/abs/2403.19444)

    使用概念瓶颈模型的ante-hoc方法将临床概念引入到分类管道中，提供了肺癌检测决策过程中的有价值见解，相较于基线深度学习模型实现了更好的分类性能（F1 > 0.9）。

    

    arXiv:2403.19444v1 公告类型：新 简要摘要：透明人工智能（XAI）领域正在迅速发展，旨在解决复杂黑匣子深度学习模型在现实应用中的信任问题。现有的事后XAI技术最近已被证明在医疗数据上表现不佳，产生不可靠的解释，不适合临床使用。为解决这一问题，我们提出了一种基于概念瓶颈模型的ante-hoc方法，首次将临床概念引入分类管道，使用户可以深入了解决策过程。在一个大型公共数据集上，我们聚焦于胸部X射线和相关医疗报告的二元分类任务，即肺癌的检测。与基准深度学习模型相比，我们的方法在肺癌检测中获得了更好的分类性能（F1 > 0.9），同时生成了临床相关且更可靠的解释。

    arXiv:2403.19444v1 Announce Type: new  Abstract: The rapidly advancing field of Explainable Artificial Intelligence (XAI) aims to tackle the issue of trust regarding the use of complex black-box deep learning models in real-world applications. Existing post-hoc XAI techniques have recently been shown to have poor performance on medical data, producing unreliable explanations which are infeasible for clinical use. To address this, we propose an ante-hoc approach based on concept bottleneck models which introduces for the first time clinical concepts into the classification pipeline, allowing the user valuable insight into the decision-making process. On a large public dataset of chest X-rays and associated medical reports, we focus on the binary classification task of lung cancer detection. Our approach yields improved classification performance in lung cancer detection when compared to baseline deep learning models (F1 > 0.9), while also generating clinically relevant and more reliable
    
[^2]: 增强鲁棒性的近似零空间增强微调方法用于视觉变换器

    Approximate Nullspace Augmented Finetuning for Robust Vision Transformers

    [https://arxiv.org/abs/2403.10476](https://arxiv.org/abs/2403.10476)

    本研究提出了一种启发自线性代数零空间概念的视觉变换器鲁棒性增强微调方法，通过合成近似零空间元素来提高模型的鲁棒性。

    

    增强深度学习模型的鲁棒性，特别是在视觉变换器（ViTs）领域中，对于它们在现实世界中的部署至关重要。在这项工作中，我们提供了一种启发自线性代数中零空间概念的视觉变换器鲁棒性增强微调方法。我们的研究集中在一个问题上，即视觉变换器是否可以展现出类似于线性映射中的零空间属性的输入变化韧性，这意味着从该零空间中采样的扰动添加到输入时不会影响模型的输出。首先，我们展示了对于许多预训练的ViTs，存在一个非平凡的零空间，这是由于存在修补嵌入层。其次，由于零空间是与线性代数相关的概念，我们表明可以利用优化策略为ViTs的非线性块合成近似零空间元素。最后，我们提出了一种细致的方法

    arXiv:2403.10476v1 Announce Type: cross  Abstract: Enhancing the robustness of deep learning models, particularly in the realm of vision transformers (ViTs), is crucial for their real-world deployment. In this work, we provide a finetuning approach to enhance the robustness of vision transformers inspired by the concept of nullspace from linear algebra. Our investigation centers on whether a vision transformer can exhibit resilience to input variations akin to the nullspace property in linear mappings, implying that perturbations sampled from this nullspace do not influence the model's output when added to the input. Firstly, we show that for many pretrained ViTs, a non-trivial nullspace exists due to the presence of the patch embedding layer. Secondly, as nullspace is a concept associated with linear algebra, we demonstrate that it is possible to synthesize approximate nullspace elements for the non-linear blocks of ViTs employing an optimisation strategy. Finally, we propose a fine-t
    

