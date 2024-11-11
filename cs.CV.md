# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Negative Yields Positive: Unified Dual-Path Adapter for Vision-Language Models](https://arxiv.org/abs/2403.12964) | 本研究在视觉-语言模型的微调中引入了双学习概念，提出了DualAdapter方法，通过正面和负面两方面的双路径适配，同时进行补充正向选择和负向排除，从而提高了在下游任务中的整体识别准确性。 |
| [^2] | [Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks](https://arxiv.org/abs/2401.17263) | 该论文提出了一种鲁棒的提示优化算法（RPO）用于对抗语言模型的破解攻击，通过梯度优化来确保输出的无害性，并成功降低了攻击成功率。 |
| [^3] | [SINCERE: Supervised Information Noise-Contrastive Estimation REvisited](https://arxiv.org/abs/2309.14277) | SINCERE提出了一个理论上合理的监督扩展，避免了同一类别的图像相互排斥，通过更好地分离不同类别的嵌入，在保持竞争性分类准确性的同时实现了更好的效果。 |
| [^4] | [Physically Realizable Natural-Looking Clothing Textures Evade Person Detectors via 3D Modeling.](http://arxiv.org/abs/2307.01778) | 通过3D建模，制作出与日常服装纹理相似的对抗性伪装纹理，可以在多个视角下避开人物检测，实现自然外观的服装纹理。 |
| [^5] | [Text-to-image Diffusion Model in Generative AI: A Survey.](http://arxiv.org/abs/2303.07909) | 本文调查了文本到图像扩散模型以及相关应用，总结了最先进的方法，并探讨了挑战和未来方向。 |

# 详细

[^1]: 负得正：用于视觉语言模型的统一双路径适配器

    Negative Yields Positive: Unified Dual-Path Adapter for Vision-Language Models

    [https://arxiv.org/abs/2403.12964](https://arxiv.org/abs/2403.12964)

    本研究在视觉-语言模型的微调中引入了双学习概念，提出了DualAdapter方法，通过正面和负面两方面的双路径适配，同时进行补充正向选择和负向排除，从而提高了在下游任务中的整体识别准确性。

    

    最近，大规模预训练的视觉-语言模型（VLMs）展示了学习开放世界视觉表示方面的巨大潜力，并通过高效微调在各种下游任务中展现出卓越性能。在这项工作中，我们创新地将双学习概念引入微调VLMs中，即我们不仅学习图像是什么，还学习图像不是什么。基于这一概念，我们提出了一种新颖的DualAdapter方法，使VLMs能够从正面和负面两方面进行双路径适配，仅使用有限的注释样本。在推理阶段，我们的DualAdapter通过针对目标类别同时进行补充正向选择和负向排除，实现了统一预测，从而提高了VLMs在下游任务中的整体识别准确性。我们广泛的实验结果跨越15个数据集，验证了所提出的DualAda

    arXiv:2403.12964v1 Announce Type: cross  Abstract: Recently, large-scale pre-trained Vision-Language Models (VLMs) have demonstrated great potential in learning open-world visual representations, and exhibit remarkable performance across a wide range of downstream tasks through efficient fine-tuning. In this work, we innovatively introduce the concept of dual learning into fine-tuning VLMs, i.e., we not only learn what an image is, but also what an image isn't. Building on this concept, we introduce a novel DualAdapter approach to enable dual-path adaptation of VLMs from both positive and negative perspectives with only limited annotated samples. In the inference stage, our DualAdapter performs unified predictions by simultaneously conducting complementary positive selection and negative exclusion across target classes, thereby enhancing the overall recognition accuracy of VLMs in downstream tasks. Our extensive experimental results across 15 datasets validate that the proposed DualAda
    
[^2]: 鲁棒的提示优化用于对抗语言模型的破解攻击

    Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks

    [https://arxiv.org/abs/2401.17263](https://arxiv.org/abs/2401.17263)

    该论文提出了一种鲁棒的提示优化算法（RPO）用于对抗语言模型的破解攻击，通过梯度优化来确保输出的无害性，并成功降低了攻击成功率。

    

    尽管在人工智能对齐方面取得了一些进展，但语言模型（LM）仍然容易受到对抗性攻击或破解攻击的影响，其中对手修改输入提示以诱导有害行为。虽然已经提出了一些防御方法，但它们仅关注狭窄的威胁模型，并不能提供强大的防御。为了实现强大的防御，我们首次提出了用于对抗破解攻击的对抗目标，并提出了一种名为鲁棒提示优化（RPO）的算法，该算法利用基于梯度的令牌优化来确保输出的无害性。通过这种方法，我们得到了一个易于访问的后缀，显著改善了对破解攻击的强韧性，包括优化过程中出现的破解攻击以及未知的破解攻击，将攻击成功率从84%降低到8.66%，在20个破解攻击中。此外，我们还发现RPO对正常LM使用的影响较小，在适应性攻击下仍然有效，并且可以迁移到黑盒模型中，降低攻击成功率。

    Despite advances in AI alignment, language models (LM) remain vulnerable to adversarial attacks or jailbreaking, in which adversaries modify input prompts to induce harmful behavior. While some defenses have been proposed, they focus on narrow threat models and fall short of a strong defense, which we posit should be effective, universal, and practical. To achieve this, we propose the first adversarial objective for defending LMs against jailbreaking attacks and an algorithm, robust prompt optimization (RPO), that uses gradient-based token optimization to enforce harmless outputs. This results in an easily accessible suffix that significantly improves robustness to both jailbreaks seen during optimization and unknown, held-out jailbreaks, reducing the attack success rate on Starling-7B from 84% to 8.66% across 20 jailbreaks. In addition, we find that RPO has a minor effect on normal LM use, is successful under adaptive attacks, and can transfer to black-box models, reducing the success
    
[^3]: SINCERE: 监督信息噪声-对比估计再审

    SINCERE: Supervised Information Noise-Contrastive Estimation REvisited

    [https://arxiv.org/abs/2309.14277](https://arxiv.org/abs/2309.14277)

    SINCERE提出了一个理论上合理的监督扩展，避免了同一类别的图像相互排斥，通过更好地分离不同类别的嵌入，在保持竞争性分类准确性的同时实现了更好的效果。

    

    信息噪声对比估计（InfoNCE）损失函数由于其强大的实证结果和理论动机，为许多自监督深度学习方法提供了基础。先前的工作表明，监督对比（SupCon）损失可扩展InfoNCE以从可用类标签中学习。然而，在这项工作中，我们发现先前的SupCon损失公式存在疑问的理由，因为它可能会促使来自同一类别的某些图像在学习到的嵌入空间中相互排斥。我们提出了监督信息噪声-对比估计再审（SINCERE）损失，作为信息噪声对比估计的理论上合理的监督扩展，它永远不会导致来自同一类别的图像相互排斥。实验表明，SINCERE导致不同类别的嵌入更好地分离，同时对于监督和迁移学习提供具有竞争力的分类准确性。我们进一步展示了一个信息论上的下界

    arXiv:2309.14277v2 Announce Type: replace-cross  Abstract: The information noise-contrastive estimation (InfoNCE) loss function provides the basis of many self-supervised deep learning methods due to its strong empirical results and theoretic motivation. Previous work suggests a supervised contrastive (SupCon) loss to extend InfoNCE to learn from available class labels. However, in this work we find that the prior SupCon loss formulation has questionable justification because it can encourage some images from the same class to repel one another in the learned embedding space. We propose the Supervised InfoNCE REvisited (SINCERE) loss as a theoretically-justified supervised extension of InfoNCE that never causes images from the same class to repel one another. Experiments show that SINCERE leads to better separation of embeddings from different classes while delivering competitive classification accuracy for supervised and transfer learning. We further show an information-theoretic boun
    
[^4]: 通过3D建模，实现自然外观的服装纹理以逃避人物检测器

    Physically Realizable Natural-Looking Clothing Textures Evade Person Detectors via 3D Modeling. (arXiv:2307.01778v1 [cs.CV])

    [http://arxiv.org/abs/2307.01778](http://arxiv.org/abs/2307.01778)

    通过3D建模，制作出与日常服装纹理相似的对抗性伪装纹理，可以在多个视角下避开人物检测，实现自然外观的服装纹理。

    

    最近的研究提出了制作对抗性服装来逃避人物检测器，但要么只对限定的视角有效，要么对人类非常明显。我们旨在基于3D建模来制作对抗性的服装纹理，这个想法已经被用于制作刚性的对抗性物体，如3D打印的乌龟。与刚性物体不同，人类和服装是非刚性的，这导致了在实际制作中的困难。为了制作出看起来自然的对抗性服装，可以在多个视角下避开人物检测器，我们提出了类似于日常服装纹理之一的对抗性伪装纹理（AdvCaT），即伪装纹理。我们利用Voronoi图和Gumbel-softmax技巧来参数化伪装纹理，并通过3D建模来优化参数。此外，我们还提出了一个高效的增强管道，将拓扑合理的投影（TopoProj）和Thin Plate Spline（TPS）结合在3D网格上使用。

    Recent works have proposed to craft adversarial clothes for evading person detectors, while they are either only effective at limited viewing angles or very conspicuous to humans. We aim to craft adversarial texture for clothes based on 3D modeling, an idea that has been used to craft rigid adversarial objects such as a 3D-printed turtle. Unlike rigid objects, humans and clothes are non-rigid, leading to difficulties in physical realization. In order to craft natural-looking adversarial clothes that can evade person detectors at multiple viewing angles, we propose adversarial camouflage textures (AdvCaT) that resemble one kind of the typical textures of daily clothes, camouflage textures. We leverage the Voronoi diagram and Gumbel-softmax trick to parameterize the camouflage textures and optimize the parameters via 3D modeling. Moreover, we propose an efficient augmentation pipeline on 3D meshes combining topologically plausible projection (TopoProj) and Thin Plate Spline (TPS) to narr
    
[^5]: 生成AI中的文本到图像扩散模型：一项调查

    Text-to-image Diffusion Model in Generative AI: A Survey. (arXiv:2303.07909v1 [cs.CV])

    [http://arxiv.org/abs/2303.07909](http://arxiv.org/abs/2303.07909)

    本文调查了文本到图像扩散模型以及相关应用，总结了最先进的方法，并探讨了挑战和未来方向。

    

    本文调查了文本到图像扩散模型，这些模型已经成为多种生成任务中流行的模型。作为一个自包含的工作，本调查从简单介绍基本扩散模型如何用于图像合成开始，接着是条件或引导如何改进学习。我们还总结了文本条件下的最先进的图像合成方法，并且进一步总结了文本引导创意生成和图像编辑的应用。除了迄今为止所取得的进展，我们还讨论了现有挑战和有前途的未来方向。

    This survey reviews text-to-image diffusion models in the context that diffusion models have emerged to be popular for a wide range of generative tasks. As a self-contained work, this survey starts with a brief introduction of how a basic diffusion model works for image synthesis, followed by how condition or guidance improves learning. Based on that, we present a review of state-of-the-art methods on text-conditioned image synthesis, i.e., text-to-image. We further summarize applications beyond text-to-image generation: text-guided creative generation and text-guided image editing. Beyond the progress made so far, we discuss existing challenges and promising future directions.
    

