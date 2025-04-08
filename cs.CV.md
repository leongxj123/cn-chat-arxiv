# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards Adversarially Robust Dataset Distillation by Curvature Regularization](https://arxiv.org/abs/2403.10045) | 本文探讨了如何通过曲率正则化方法在精炼数据集中嵌入对抗鲁棒性，以保持模型高准确性并获得更好的对抗鲁棒性。 |
| [^2] | [Backdoor Defense with Non-Adversarial Backdoor.](http://arxiv.org/abs/2307.15539) | 提出了一种非对抗性后门防御框架，通过在被污染样本中注入非对抗性后门，当触发时可以抑制攻击者对污染数据的后门攻击，同时保持对干净数据的影响有限。 |
| [^3] | [The detection and rectification for identity-switch based on unfalsified control.](http://arxiv.org/abs/2307.14591) | 本文提出了一种基于未被伪造的控制的多目标跟踪方法，针对身份交换问题设计了检测和修正模块，以及解决外观信息模糊匹配的策略，并在实验中展示了其出色的效果和鲁棒性。 |
| [^4] | [InceptionNeXt: When Inception Meets ConvNeXt.](http://arxiv.org/abs/2303.16900) | 本论文提出了一种名为InceptionNeXt的新型神经网络，通过将大内核卷积沿通道维度分解为四个平行分支来提高模型效率，解决了保持性能的同时加快基于大内核的CNN模型的问题。 |

# 详细

[^1]: 通过曲率正则化实现对抗鲁棒性数据集精炼

    Towards Adversarially Robust Dataset Distillation by Curvature Regularization

    [https://arxiv.org/abs/2403.10045](https://arxiv.org/abs/2403.10045)

    本文探讨了如何通过曲率正则化方法在精炼数据集中嵌入对抗鲁棒性，以保持模型高准确性并获得更好的对抗鲁棒性。

    

    数据集精炼（DD）允许将数据集精炼为原始大小的分数，同时保留丰富的分布信息，使得在精炼数据集上训练的模型可以在节省显著计算负载的同时达到可比的准确性。最近在这一领域的研究集中在提高在精炼数据集上训练的模型的准确性。在本文中，我们旨在探索DD的一种新视角。我们研究如何在精炼数据集中嵌入对抗鲁棒性，以使在这些数据集上训练的模型保持高精度的同时获得更好的对抗鲁棒性。我们提出了一种通过将曲率正则化纳入到精炼过程中来实现这一目标的新方法，而这种方法的计算开销比标准的对抗训练要少得多。大量的实证实验表明，我们的方法不仅在准确性上优于标准对抗训练，同时在对抗性能方面也取得了显著改进。

    arXiv:2403.10045v1 Announce Type: new  Abstract: Dataset distillation (DD) allows datasets to be distilled to fractions of their original size while preserving the rich distributional information so that models trained on the distilled datasets can achieve a comparable accuracy while saving significant computational loads. Recent research in this area has been focusing on improving the accuracy of models trained on distilled datasets. In this paper, we aim to explore a new perspective of DD. We study how to embed adversarial robustness in distilled datasets, so that models trained on these datasets maintain the high accuracy and meanwhile acquire better adversarial robustness. We propose a new method that achieves this goal by incorporating curvature regularization into the distillation process with much less computational overhead than standard adversarial training. Extensive empirical experiments suggest that our method not only outperforms standard adversarial training on both accur
    
[^2]: 非对抗性后门防御

    Backdoor Defense with Non-Adversarial Backdoor. (arXiv:2307.15539v1 [cs.LG])

    [http://arxiv.org/abs/2307.15539](http://arxiv.org/abs/2307.15539)

    提出了一种非对抗性后门防御框架，通过在被污染样本中注入非对抗性后门，当触发时可以抑制攻击者对污染数据的后门攻击，同时保持对干净数据的影响有限。

    

    深度神经网络（DNNs）容易受到后门攻击的影响，这种攻击并不会影响网络对干净数据的性能，但一旦添加触发模式，就会操纵网络行为。现有的防御方法大大降低了攻击成功率，但它们在干净数据上的预测准确性仍然远远落后于干净模型。受后门攻击的隐蔽性和有效性的启发，我们提出了一个简单但非常有效的防御框架，该框架注入了针对被污染样本的非对抗性后门。按照后门攻击的一般步骤，我们检测一小组可疑样本，然后对它们应用毒化策略。一旦触发，非对抗性后门抑制了攻击者对污染数据的后门攻击，但对干净数据的影响有限。防御可以在数据预处理期间进行，而不需要对标准的端到端训练流程进行任何修改。

    Deep neural networks (DNNs) are vulnerable to backdoor attack, which does not affect the network's performance on clean data but would manipulate the network behavior once a trigger pattern is added. Existing defense methods have greatly reduced attack success rate, but their prediction accuracy on clean data still lags behind a clean model by a large margin. Inspired by the stealthiness and effectiveness of backdoor attack, we propose a simple but highly effective defense framework which injects non-adversarial backdoors targeting poisoned samples. Following the general steps in backdoor attack, we detect a small set of suspected samples and then apply a poisoning strategy to them. The non-adversarial backdoor, once triggered, suppresses the attacker's backdoor on poisoned data, but has limited influence on clean data. The defense can be carried out during data preprocessing, without any modification to the standard end-to-end training pipeline. We conduct extensive experiments on mul
    
[^3]: 基于未被伪造的控制的身份交换检测与修正

    The detection and rectification for identity-switch based on unfalsified control. (arXiv:2307.14591v1 [cs.CV])

    [http://arxiv.org/abs/2307.14591](http://arxiv.org/abs/2307.14591)

    本文提出了一种基于未被伪造的控制的多目标跟踪方法，针对身份交换问题设计了检测和修正模块，以及解决外观信息模糊匹配的策略，并在实验中展示了其出色的效果和鲁棒性。

    

    多目标跟踪的目的是持续跟踪和识别视频中检测到的物体。目前，大多数多目标跟踪方法都是通过建模运动信息并将其与外观信息相结合，来确定和跟踪物体。本文采用了未被伪造的控制方法来解决多目标跟踪中的身份交换问题。我们在跟踪过程中建立了外观信息变化的序列，针对身份交换检测和恢复设计了一个检测和修正模块。我们还提出了一种简单而有效的策略来解决数据关联过程中外观信息模糊匹配的问题。公开可用的多目标跟踪数据集上的实验结果表明，该跟踪器在处理由遮挡和快速运动引起的跟踪错误方面具有出色的效果和鲁棒性。

    The purpose of multi-object tracking (MOT) is to continuously track and identify objects detected in videos. Currently, most methods for multi-object tracking model the motion information and combine it with appearance information to determine and track objects. In this paper, unfalsified control is employed to address the ID-switch problem in multi-object tracking. We establish sequences of appearance information variations for the trajectories during the tracking process and design a detection and rectification module specifically for ID-switch detection and recovery. We also propose a simple and effective strategy to address the issue of ambiguous matching of appearance information during the data association process. Experimental results on publicly available MOT datasets demonstrate that the tracker exhibits excellent effectiveness and robustness in handling tracking errors caused by occlusions and rapid movements.
    
[^4]: InceptionNeXt：当Inception遇到ConvNeXt

    InceptionNeXt: When Inception Meets ConvNeXt. (arXiv:2303.16900v1 [cs.CV])

    [http://arxiv.org/abs/2303.16900](http://arxiv.org/abs/2303.16900)

    本论文提出了一种名为InceptionNeXt的新型神经网络，通过将大内核卷积沿通道维度分解为四个平行分支来提高模型效率，解决了保持性能的同时加快基于大内核的CNN模型的问题。

    

    受ViTs长程建模能力的启发，近期广泛研究和采用了大内核卷积来扩大感受野和提高模型性能，例如ConvNeXt采用了7x7深度卷积。虽然这种深度操作仅消耗少量FLOPs，但由于高内存访问成本，这在功能强大的计算设备上大大损害了模型效率。尽管缩小ConvNeXt的内核大小能提高速度，但会导致性能显着下降。如何在保持性能的同时加快基于大内核的CNN模型仍不清楚。为了解决这个问题，受Inceptions的启发，我们提出将大内核深度卷积沿通道维度分解为四个平行分支，即小方内核、两个正交带内核和一个互补内核。

    Inspired by the long-range modeling ability of ViTs, large-kernel convolutions are widely studied and adopted recently to enlarge the receptive field and improve model performance, like the remarkable work ConvNeXt which employs 7x7 depthwise convolution. Although such depthwise operator only consumes a few FLOPs, it largely harms the model efficiency on powerful computing devices due to the high memory access costs. For example, ConvNeXt-T has similar FLOPs with ResNet-50 but only achieves 60% throughputs when trained on A100 GPUs with full precision. Although reducing the kernel size of ConvNeXt can improve speed, it results in significant performance degradation. It is still unclear how to speed up large-kernel-based CNN models while preserving their performance. To tackle this issue, inspired by Inceptions, we propose to decompose large-kernel depthwise convolution into four parallel branches along channel dimension, i.e. small square kernel, two orthogonal band kernels, and an ide
    

