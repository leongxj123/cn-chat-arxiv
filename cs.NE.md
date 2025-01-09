# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NeuralDiffuser: Controllable fMRI Reconstruction with Primary Visual Feature Guided Diffusion](https://arxiv.org/abs/2402.13809) | NeuralDiffuser引入主视觉特征引导，扩展了LDM方法的自下而上过程，以实现忠实的语义和细节。 |

# 详细

[^1]: NeuralDiffuser：具有主视觉特征引导扩散的可控fMRI重建

    NeuralDiffuser: Controllable fMRI Reconstruction with Primary Visual Feature Guided Diffusion

    [https://arxiv.org/abs/2402.13809](https://arxiv.org/abs/2402.13809)

    NeuralDiffuser引入主视觉特征引导，扩展了LDM方法的自下而上过程，以实现忠实的语义和细节。

    

    基于潜在扩散模型(LDM)从功能性磁共振成像(fMRI)中重建视觉刺激，为大脑提供了细粒度的检索。一个挑战在于重建细节的连贯对齐（如结构、背景、纹理、颜色等）。此外，即使在相同条件下，LDM也会生成不同的图像结果。因此，我们首先揭示了基于LDM的神经科学视角，即基于来自海量图像的预训练知识进行自上而下的创建，但缺乏基于细节驱动的自下而上感知，导致细节不忠实。我们提出了NeuralDiffuser，引入主视觉特征引导，以渐变形式提供细节线索，扩展了LDM方法的自下而上过程，以实现忠实的语义和细节。我们还开发了一种新颖的引导策略，以确保重复重建的一致性，而不是随机性。

    arXiv:2402.13809v1 Announce Type: cross  Abstract: Reconstructing visual stimuli from functional Magnetic Resonance Imaging (fMRI) based on Latent Diffusion Models (LDM) provides a fine-grained retrieval of the brain. A challenge persists in reconstructing a cohesive alignment of details (such as structure, background, texture, color, etc.). Moreover, LDMs would generate different image results even under the same conditions. For these, we first uncover the neuroscientific perspective of LDM-based methods that is top-down creation based on pre-trained knowledge from massive images but lack of detail-driven bottom-up perception resulting in unfaithful details. We propose NeuralDiffuser which introduces primary visual feature guidance to provide detail cues in the form of gradients, extending the bottom-up process for LDM-based methods to achieve faithful semantics and details. We also developed a novel guidance strategy to ensure the consistency of repeated reconstructions rather than a
    

