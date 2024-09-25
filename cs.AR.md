# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MicroT: Low-Energy and Adaptive Models for MCUs](https://arxiv.org/abs/2403.08040) | MicroT是一个低能耗、多任务自适应模型框架，通过特征提取器和分类器的分离、模型优化和本地任务训练，在MCUs上实现了模型性能的提升和能耗的降低。 |

# 详细

[^1]: MicroT：用于MCUs的低能耗和自适应模型

    MicroT: Low-Energy and Adaptive Models for MCUs

    [https://arxiv.org/abs/2403.08040](https://arxiv.org/abs/2403.08040)

    MicroT是一个低能耗、多任务自适应模型框架，通过特征提取器和分类器的分离、模型优化和本地任务训练，在MCUs上实现了模型性能的提升和能耗的降低。

    

    我们提出了MicroT，这是一个面向资源受限的MCUs的低能耗、多任务自适应模型框架。我们将原始模型划分为特征提取器和分类器。特征提取器通过自监督知识蒸馏获得，并通过模型分割和联合训练进一步优化为部分模型和完整模型。然后将这些模型部署在MCUs上，增加并在本地任务上训练分类器，最终执行关节推理的阶段决策。在这个过程中，部分模型最初处理样本，如果置信度得分低于设定的阈值，完整模型将恢复并继续推理。我们在两个模型、三个数据集和两个MCU板上评估了MicroT。我们的实验评估表明，在处理多个本地任务时，MicroT有效地提高了模型性能并降低了能耗。与未经优化的特征提取器相比，MicroT

    arXiv:2403.08040v1 Announce Type: new  Abstract: We propose MicroT, a low-energy, multi-task adaptive model framework for resource-constrained MCUs. We divide the original model into a feature extractor and a classifier. The feature extractor is obtained through self-supervised knowledge distillation and further optimized into part and full models through model splitting and joint training. These models are then deployed on MCUs, with classifiers added and trained on local tasks, ultimately performing stage-decision for joint inference. In this process, the part model initially processes the sample, and if the confidence score falls below the set threshold, the full model will resume and continue the inference. We evaluate MicroT on two models, three datasets, and two MCU boards. Our experimental evaluation shows that MicroT effectively improves model performance and reduces energy consumption when dealing with multiple local tasks. Compared to the unoptimized feature extractor, MicroT
    

