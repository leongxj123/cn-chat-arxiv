# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rethinking Low-quality Optical Flow in Unsupervised Surgical Instrument Segmentation](https://arxiv.org/abs/2403.10039) | 本研究在无监督手术器械分割中解决了由于低质量光流而引起的挑战，提出了一种三重策略：直接从光流中提取边界、选择性丢弃质量较差的帧、以及利用可变帧率进行微调。在数据集上进行了充分评估，展示出有前景的结果。 |
| [^2] | [Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey](https://arxiv.org/abs/2402.02242) | 本综述调研了面向预训练视觉模型的参数高效微调方法，通过最小参数修改超越全面微调的性能，提供了全面的概述和未来方向，并提供了丰富的资源收藏。 |
| [^3] | [Towards Real-World Test-Time Adaptation: Tri-Net Self-Training with Balanced Normalization.](http://arxiv.org/abs/2309.14949) | 本文研究了真实世界测试时间自适应问题，在全局类别不平衡的测试集上补充了现有的协议，并提出了一种平衡归一化层来适应不平衡的测试数据，以解决现有方法的失败。 |
| [^4] | [Generalizable Memory-driven Transformer for Multivariate Long Sequence Time-series Forecasting.](http://arxiv.org/abs/2207.07827) | 本文提出了一种通用记忆驱动变压器，通过集成多个时间序列特征来驱动预测过程，逐步引入噪声以增强泛化能力，在多个数据集上实现了更优秀的预测性能。 |

# 详细

[^1]: 重新思考无监督手术器械分割中低质量光流问题

    Rethinking Low-quality Optical Flow in Unsupervised Surgical Instrument Segmentation

    [https://arxiv.org/abs/2403.10039](https://arxiv.org/abs/2403.10039)

    本研究在无监督手术器械分割中解决了由于低质量光流而引起的挑战，提出了一种三重策略：直接从光流中提取边界、选择性丢弃质量较差的帧、以及利用可变帧率进行微调。在数据集上进行了充分评估，展示出有前景的结果。

    

    视频的手术器械分割在机器人辅助手术中扮演着重要角色。与监督设置不同，无监督分割主要依赖于运动线索，然而由于手术镜头中光流通常比自然场景中的要低质量，这些运动线索很难识别。本研究致力于解决即使面对低质量光流固有限制，提高模型性能的挑战。我们的方法从三个方面入手：直接从光流中提取边界、有选择地丢弃质量较差的帧、以及利用可变帧率的微调过程。我们在EndoVis2017 VOS数据集和Endovis2017挑战数据集上对我们的策略进行了彻底评估，模型展现出有前景的结果，实现了均值交叉。

    arXiv:2403.10039v1 Announce Type: cross  Abstract: Video-based surgical instrument segmentation plays an important role in robot-assisted surgeries. Unlike supervised settings, unsupervised segmentation relies heavily on motion cues, which are challenging to discern due to the typically lower quality of optical flow in surgical footage compared to natural scenes. This presents a considerable burden for the advancement of unsupervised segmentation techniques. In our work, we address the challenge of enhancing model performance despite the inherent limitations of low-quality optical flow. Our methodology employs a three-pronged approach: extracting boundaries directly from the optical flow, selectively discarding frames with inferior flow quality, and employing a fine-tuning process with variable frame rates. We thoroughly evaluate our strategy on the EndoVis2017 VOS dataset and Endovis2017 Challenge dataset, where our model demonstrates promising results, achieving a mean Intersection-o
    
[^2]: 面向预训练视觉模型的参数高效微调：一项综述

    Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey

    [https://arxiv.org/abs/2402.02242](https://arxiv.org/abs/2402.02242)

    本综述调研了面向预训练视觉模型的参数高效微调方法，通过最小参数修改超越全面微调的性能，提供了全面的概述和未来方向，并提供了丰富的资源收藏。

    

    大规模预训练的视觉模型（PVMs）展示了在各种下游视觉任务中的适应能力潜力。然而，随着最先进的PVMs达到数十亿甚至数万亿个参数，标准的全面微调范式由于高计算和存储需求变得不可持续。作为响应，研究人员正在探索参数高效微调（PEFT），旨在以最小参数修改超越全面微调的性能。本综述提供了视觉PEFT的全面概述和未来方向，对最新进展进行了系统审查。首先，我们提供了PEFT的正式定义，并讨论了模型预训练方法。然后，我们将现有方法分为三类：基于添加的、基于部分的和基于统一的。最后，我们介绍了常用的数据集和应用，并提出了潜在的未来研究挑战。该综述还提供了丰富的资源收藏。

    Large-scale pre-trained vision models (PVMs) have shown great potential for adaptability across various downstream vision tasks. However, with state-of-the-art PVMs growing to billions or even trillions of parameters, the standard full fine-tuning paradigm is becoming unsustainable due to high computational and storage demands. In response, researchers are exploring parameter-efficient fine-tuning (PEFT), which seeks to exceed the performance of full fine-tuning with minimal parameter modifications. This survey provides a comprehensive overview and future directions for visual PEFT, offering a systematic review of the latest advancements. First, we provide a formal definition of PEFT and discuss model pre-training methods. We then categorize existing methods into three categories: addition-based, partial-based, and unified-based. Finally, we introduce the commonly used datasets and applications and suggest potential future research challenges. A comprehensive collection of resources is
    
[^3]: 面向真实世界测试时间自适应：具有平衡归一化的三网络自训练

    Towards Real-World Test-Time Adaptation: Tri-Net Self-Training with Balanced Normalization. (arXiv:2309.14949v1 [cs.LG])

    [http://arxiv.org/abs/2309.14949](http://arxiv.org/abs/2309.14949)

    本文研究了真实世界测试时间自适应问题，在全局类别不平衡的测试集上补充了现有的协议，并提出了一种平衡归一化层来适应不平衡的测试数据，以解决现有方法的失败。

    

    测试时间自适应旨在将源域模型适应到推断阶段的测试数据中，在适应到未见过的破损情况下取得了成功。然而，在更具挑战性的真实世界情境下，这些尝试可能会失败。现有的研究主要考虑非独立同分布的数据流和持续的领域转移下的真实世界测试时间自适应。在这项工作中，我们首先用全局类别不平衡的测试集来补充现有的真实世界TTA协议。我们证明把所有设置结合起来对现有方法提出了新的挑战。我们认为最先失败的现有方法是因为不加选择地将归一化层适应到不平衡的测试数据上所导致的。为了解决这个缺点，我们提出了一个平衡批量归一化层，在推断阶段替换原来的批量归一化。新的批量归一化层能够适应而不偏向多数类别。我们受到自学习（ST）在无标签学习中的成功启发。

    Test-Time Adaptation aims to adapt source domain model to testing data at inference stage with success demonstrated in adapting to unseen corruptions. However, these attempts may fail under more challenging real-world scenarios. Existing works mainly consider real-world test-time adaptation under non-i.i.d. data stream and continual domain shift. In this work, we first complement the existing real-world TTA protocol with a globally class imbalanced testing set. We demonstrate that combining all settings together poses new challenges to existing methods. We argue the failure of state-of-the-art methods is first caused by indiscriminately adapting normalization layers to imbalanced testing data. To remedy this shortcoming, we propose a balanced batchnorm layer to swap out the regular batchnorm at inference stage. The new batchnorm layer is capable of adapting without biasing towards majority classes. We are further inspired by the success of self-training~(ST) in learning from unlabeled 
    
[^4]: 多元长序列时间序列预测的通用记忆驱动变压器

    Generalizable Memory-driven Transformer for Multivariate Long Sequence Time-series Forecasting. (arXiv:2207.07827v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2207.07827](http://arxiv.org/abs/2207.07827)

    本文提出了一种通用记忆驱动变压器，通过集成多个时间序列特征来驱动预测过程，逐步引入噪声以增强泛化能力，在多个数据集上实现了更优秀的预测性能。

    

    多元长序列时间序列预测(M-LSTF)是一个实际但具有挑战性的问题。与传统的时间序列预测任务不同，M-LSTF任务从两个方面更具挑战性：1) M-LSTF模型需要同时学习多个时间特征之间的时间序列模式；2)在滚动预测设置中，两个连续训练样本之间的相似度随着预测长度的增加而增加，这使得模型更易于过拟合。本文提出了一种通用记忆驱动变压器来解决M-LSTF问题。具体而言，我们首先提出了一个全局层面的记忆组件，通过集成多个时间序列特征来驱动预测过程。此外，我们采用渐进式的方式来训练我们的模型，以增强其泛化能力，逐步在训练样本中引入伯努利噪声。在多个领域的五个不同数据集上进行了大量实验。实验结果表明，我们提出的模型优于现有的方法，并在所有数据集上实现了更优异的预测性能。

    Multivariate long sequence time-series forecasting (M-LSTF) is a practical but challenging problem. Unlike traditional timer-series forecasting tasks, M-LSTF tasks are more challenging from two aspects: 1) M-LSTF models need to learn time-series patterns both within and between multiple time features; 2) Under the rolling forecasting setting, the similarity between two consecutive training samples increases with the increasing prediction length, which makes models more prone to overfitting. In this paper, we propose a generalizable memory-driven Transformer to target M-LSTF problems. Specifically, we first propose a global-level memory component to drive the forecasting procedure by integrating multiple time-series features. In addition, we adopt a progressive fashion to train our model to increase its generalizability, in which we gradually introduce Bernoulli noises to training samples. Extensive experiments have been performed on five different datasets across multiple fields. Exper
    

