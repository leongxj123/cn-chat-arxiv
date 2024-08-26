# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Exploring Multi-modal Neural Scene Representations With Applications on Thermal Imaging](https://arxiv.org/abs/2403.11865) | 本文对神经场景表示在多模态学习中的应用进行了全面评估，并提出了四种不同的策略以将第二模态（非RGB）纳入NeRFs中，通过选择热成像作为第二模态来挑战神经场景表示的整合。 |

# 详细

[^1]: 利用热成像探索多模态神经场景表示并应用

    Exploring Multi-modal Neural Scene Representations With Applications on Thermal Imaging

    [https://arxiv.org/abs/2403.11865](https://arxiv.org/abs/2403.11865)

    本文对神经场景表示在多模态学习中的应用进行了全面评估，并提出了四种不同的策略以将第二模态（非RGB）纳入NeRFs中，通过选择热成像作为第二模态来挑战神经场景表示的整合。

    

    神经辐射场（NeRFs）在一组RGB图像上训练时迅速发展为新的事实标准，用于新视角合成任务。本文在多模态学习的背景下对神经场景表示（如NeRFs）进行了全面评估。具体而言，我们提出了四种不同的策略，用于如何将第二模态（非RGB）纳入NeRFs中：（1）独立地从头训练每种模态；（2）在RGB上进行预训练，然后在第二模态上进行微调；（3）添加第二分支；（4）添加一个单独的组件来预测（颜色）额外模态的值。我们选择了热成像作为第二模态，因为从辐射度来看，它与RGB有很大差异，这使得将其整合到神经场景表示中具有挑战性。

    arXiv:2403.11865v1 Announce Type: cross  Abstract: Neural Radiance Fields (NeRFs) quickly evolved as the new de-facto standard for the task of novel view synthesis when trained on a set of RGB images. In this paper, we conduct a comprehensive evaluation of neural scene representations, such as NeRFs, in the context of multi-modal learning. Specifically, we present four different strategies of how to incorporate a second modality, other than RGB, into NeRFs: (1) training from scratch independently on both modalities; (2) pre-training on RGB and fine-tuning on the second modality; (3) adding a second branch; and (4) adding a separate component to predict (color) values of the additional modality. We chose thermal imaging as second modality since it strongly differs from RGB in terms of radiosity, making it challenging to integrate into neural scene representations. For the evaluation of the proposed strategies, we captured a new publicly available multi-view dataset, ThermalMix, consisti
    

