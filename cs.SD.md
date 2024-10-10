# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Data-Driven Room Acoustic Modeling Via Differentiable Feedback Delay Networks With Learnable Delay Lines](https://arxiv.org/abs/2404.00082) | 通过可学习延迟线实现可微分反馈延迟网络的参数优化，实现了对室内声学特性的数据驱动建模。 |
| [^2] | [Speech Separation based on Contrastive Learning and Deep Modularization.](http://arxiv.org/abs/2305.10652) | 本文提出了一种基于对比学习和深度模块化的完全无监督语音分离方法，解决了有监督学习中存在的排列问题、说话人数量不匹配的问题和高质量标记数据的依赖问题。 |

# 详细

[^1]: 基于可微分反馈延迟网络和可学习延迟线的数据驱动室内声学建模

    Data-Driven Room Acoustic Modeling Via Differentiable Feedback Delay Networks With Learnable Delay Lines

    [https://arxiv.org/abs/2404.00082](https://arxiv.org/abs/2404.00082)

    通过可学习延迟线实现可微分反馈延迟网络的参数优化，实现了对室内声学特性的数据驱动建模。

    

    在过去的几十年中，人们致力于设计人工混响算法，旨在模拟物理环境的室内声学。尽管取得了显著进展，但延迟网络模型的自动参数调整仍然是一个开放性挑战。我们提出了一种新方法，通过学习可微分反馈延迟网络（FDN）的参数，使其输出呈现出所测得的室内脉冲响应的感知特性。

    arXiv:2404.00082v1 Announce Type: cross  Abstract: Over the past few decades, extensive research has been devoted to the design of artificial reverberation algorithms aimed at emulating the room acoustics of physical environments. Despite significant advancements, automatic parameter tuning of delay-network models remains an open challenge. We introduce a novel method for finding the parameters of a Feedback Delay Network (FDN) such that its output renders the perceptual qualities of a measured room impulse response. The proposed approach involves the implementation of a differentiable FDN with trainable delay lines, which, for the first time, allows us to simultaneously learn each and every delay-network parameter via backpropagation. The iterative optimization process seeks to minimize a time-domain loss function incorporating differentiable terms accounting for energy decay and echo density. Through experimental validation, we show that the proposed method yields time-invariant freq
    
[^2]: 基于对比学习和深度模块化的语音分离

    Speech Separation based on Contrastive Learning and Deep Modularization. (arXiv:2305.10652v1 [cs.SD])

    [http://arxiv.org/abs/2305.10652](http://arxiv.org/abs/2305.10652)

    本文提出了一种基于对比学习和深度模块化的完全无监督语音分离方法，解决了有监督学习中存在的排列问题、说话人数量不匹配的问题和高质量标记数据的依赖问题。

    

    目前，语音分离的最先进工具依赖于有监督学习。这意味着它们必须处理排列问题，它们受到训练和推断中使用的说话者数量不匹配的影响。此外，它们的性能严重依赖于高质量标记数据的存在。这些问题可以通过采用完全无监督的语音分离技术有效地解决。在本文中，我们使用对比学习建立帧的表示，然后在下游的深度模块化任务中使用学习到的表示。具体而言，在语音分离中，说话人的不同帧可以被看作是给定那个说话人的隐含标准帧的增强版。说话人的帧包含足够的韵律信息重叠，这是语音分离的关键。基于此，我们实现了自监督学习，学习缩小帧之间的距离。

    The current monaural state of the art tools for speech separation relies on supervised learning. This means that they must deal with permutation problem, they are impacted by the mismatch on the number of speakers used in training and inference. Moreover, their performance heavily relies on the presence of high-quality labelled data. These problems can be effectively addressed by employing a fully unsupervised technique for speech separation. In this paper, we use contrastive learning to establish the representations of frames then use the learned representations in the downstream deep modularization task. Concretely, we demonstrate experimentally that in speech separation, different frames of a speaker can be viewed as augmentations of a given hidden standard frame of that speaker. The frames of a speaker contain enough prosodic information overlap which is key in speech separation. Based on this, we implement a self-supervised learning to learn to minimize the distance between frames
    

