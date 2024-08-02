# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Ultra-High-Resolution Detector Simulation with Intra-Event Aware GAN and Self-Supervised Relational Reasoning.](http://arxiv.org/abs/2303.08046) | 本文提出了一种新颖的探测器模拟方法IEA-GAN，通过产生与图层相关的上下文化的图像，提高了超高分辨率探测器响应的相关性和多样性。同时，引入新的事件感知损失和统一性损失，显著提高了图像的保真度和多样性。 |

# 详细

[^1]: 基于事件感知的生成对抗网络和自监督关系推理的超高分辨率探测器模拟

    Ultra-High-Resolution Detector Simulation with Intra-Event Aware GAN and Self-Supervised Relational Reasoning. (arXiv:2303.08046v1 [physics.ins-det])

    [http://arxiv.org/abs/2303.08046](http://arxiv.org/abs/2303.08046)

    本文提出了一种新颖的探测器模拟方法IEA-GAN，通过产生与图层相关的上下文化的图像，提高了超高分辨率探测器响应的相关性和多样性。同时，引入新的事件感知损失和统一性损失，显著提高了图像的保真度和多样性。

    

    在粒子物理学中，模拟高分辨率探测器响应一直是一个存储成本高、计算密集的过程。尽管深度生成模型可以使这个过程更具成本效益，但超高分辨率探测器模拟仍然很困难，因为它包含了事件内相关和细粒度的相互信息。为了克服这些限制，我们提出了一种新颖的生成对抗网络方法（IEA-GAN），融合了自监督学习和关系推理模型。IEA-GAN提出了一个关系推理模块，近似于探测器模拟中“事件”的概念，可以生成与图层相关的上下文化的图像，提高了超高分辨率探测器响应的相关性和多样性。IEA-GAN还引入了新的事件感知损失和统一性损失，显著提高了图像的保真度和多样性。我们展示了IEA-GAN的应用。

    Simulating high-resolution detector responses is a storage-costly and computationally intensive process that has long been challenging in particle physics. Despite the ability of deep generative models to make this process more cost-efficient, ultra-high-resolution detector simulation still proves to be difficult as it contains correlated and fine-grained mutual information within an event. To overcome these limitations, we propose Intra-Event Aware GAN (IEA-GAN), a novel fusion of Self-Supervised Learning and Generative Adversarial Networks. IEA-GAN presents a Relational Reasoning Module that approximates the concept of an ''event'' in detector simulation, allowing for the generation of correlated layer-dependent contextualized images for high-resolution detector responses with a proper relational inductive bias. IEA-GAN also introduces a new intra-event aware loss and a Uniformity loss, resulting in significant enhancements to image fidelity and diversity. We demonstrate IEA-GAN's ap
    

