# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AICAttack: Adversarial Image Captioning Attack with Attention-Based Optimization](https://arxiv.org/abs/2402.11940) | 提出了一种新的对抗攻击策略AICAttack，旨在通过微小的图像扰动来攻击图像字幕模型，在黑盒攻击情景下具有良好的效果。 |

# 详细

[^1]: AICAttack：基于注意力优化的对抗性图像字幕攻击

    AICAttack: Adversarial Image Captioning Attack with Attention-Based Optimization

    [https://arxiv.org/abs/2402.11940](https://arxiv.org/abs/2402.11940)

    提出了一种新的对抗攻击策略AICAttack，旨在通过微小的图像扰动来攻击图像字幕模型，在黑盒攻击情景下具有良好的效果。

    

    最近深度学习研究取得了在计算机视觉（CV）和自然语言处理（NLP）等许多任务上显著的成就。CV和NLP交叉点上的图像字幕问题中，相关模型对抗攻击的稳健性尚未得到充分研究。本文提出了一种新颖的对抗攻击策略，称为AICAttack（基于注意力的图像字幕攻击），旨在通过对图像进行微小扰动来攻击图像字幕模型。在黑盒攻击环境中运行，我们的算法不需要访问目标模型的架构、参数或梯度信息。我们引入了基于注意力的候选选择机制，可识别最佳像素进行攻击，然后采用差分进化（DE）来扰乱像素的RGB值。通过对基准上的广泛实验，我们证明了AICAttack的有效性。

    arXiv:2402.11940v1 Announce Type: cross  Abstract: Recent advances in deep learning research have shown remarkable achievements across many tasks in computer vision (CV) and natural language processing (NLP). At the intersection of CV and NLP is the problem of image captioning, where the related models' robustness against adversarial attacks has not been well studied. In this paper, we present a novel adversarial attack strategy, which we call AICAttack (Attention-based Image Captioning Attack), designed to attack image captioning models through subtle perturbations on images. Operating within a black-box attack scenario, our algorithm requires no access to the target model's architecture, parameters, or gradient information. We introduce an attention-based candidate selection mechanism that identifies the optimal pixels to attack, followed by Differential Evolution (DE) for perturbing pixels' RGB values. We demonstrate AICAttack's effectiveness through extensive experiments on benchma
    

