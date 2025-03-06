# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Assessing Robustness via Score-Based Adversarial Image Generation.](http://arxiv.org/abs/2310.04285) | 本论文介绍了一种基于分数的对抗生成框架（ScoreAG），可以生成超过$\ell_p$-范数约束的对抗性示例，并通过图像转换或新图像合成的方法保持图像的核心语义，大大增强了分类器的鲁棒性。 |
| [^2] | [HoSNN: Adversarially-Robust Homeostatic Spiking Neural Networks with Adaptive Firing Thresholds.](http://arxiv.org/abs/2308.10373) | HoSNN是一种对抗性稳态脉冲神经网络，通过采用自适应发放阈值的渗漏整合与发放（TA-LIF）神经元模型来抵御对抗攻击，并在无监督的方式下保护其鲁棒性。 |

# 详细

[^1]: 通过基于分数的对抗图像生成评估鲁棒性

    Assessing Robustness via Score-Based Adversarial Image Generation. (arXiv:2310.04285v1 [cs.CV])

    [http://arxiv.org/abs/2310.04285](http://arxiv.org/abs/2310.04285)

    本论文介绍了一种基于分数的对抗生成框架（ScoreAG），可以生成超过$\ell_p$-范数约束的对抗性示例，并通过图像转换或新图像合成的方法保持图像的核心语义，大大增强了分类器的鲁棒性。

    

    大多数对抗攻击和防御都集中在小的$\ell_p$-范数约束内的扰动上。然而，$\ell_p$威胁模型无法捕捉到所有相关的保留语义的扰动，因此，鲁棒性评估的范围是有限的。在这项工作中，我们引入了基于分数的对抗生成（ScoreAG），一种利用基于分数的生成模型的进展来生成超过$\ell_p$-范数约束的对抗性示例的新的框架，称为无限制的对抗性示例，克服了它们的局限性。与传统方法不同，ScoreAG在生成逼真的对抗性示例时保持图像的核心语义，可以通过转换现有图像或完全从零开始合成新图像的方式实现。我们进一步利用ScoreAG的生成能力来净化图像，从经验上增强分类器的鲁棒性。我们的大量实证评估表明，ScoreAG与现有最先进的对抗攻击方法的性能相当。

    Most adversarial attacks and defenses focus on perturbations within small $\ell_p$-norm constraints. However, $\ell_p$ threat models cannot capture all relevant semantic-preserving perturbations, and hence, the scope of robustness evaluations is limited. In this work, we introduce Score-Based Adversarial Generation (ScoreAG), a novel framework that leverages the advancements in score-based generative models to generate adversarial examples beyond $\ell_p$-norm constraints, so-called unrestricted adversarial examples, overcoming their limitations. Unlike traditional methods, ScoreAG maintains the core semantics of images while generating realistic adversarial examples, either by transforming existing images or synthesizing new ones entirely from scratch. We further exploit the generative capability of ScoreAG to purify images, empirically enhancing the robustness of classifiers. Our extensive empirical evaluation demonstrates that ScoreAG matches the performance of state-of-the-art atta
    
[^2]: HoSNN: 具有自适应发放阈值的对抗性稳态脉冲神经网络

    HoSNN: Adversarially-Robust Homeostatic Spiking Neural Networks with Adaptive Firing Thresholds. (arXiv:2308.10373v2 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2308.10373](http://arxiv.org/abs/2308.10373)

    HoSNN是一种对抗性稳态脉冲神经网络，通过采用自适应发放阈值的渗漏整合与发放（TA-LIF）神经元模型来抵御对抗攻击，并在无监督的方式下保护其鲁棒性。

    

    脉冲神经网络（SNNs）在高效和强大的神经启发式计算方面具有潜力。然而，与其他类型的神经网络一样，SNNs面临着对抗攻击的严重问题。我们提出了第一个从神经恒稳性中汲取灵感的研究，以开发一种仿生解决方案，来应对SNNs对对抗性攻击的敏感性。我们的方法的核心是一种新颖的自适应发放阈值的渗漏整合与发放（TA-LIF）神经元模型，我们采用它来构建所提出的对抗性稳态SNN（HoSNN）。与传统的LIF模型不同，我们的TA-LIF模型融入了自稳定动态阈值机制，限制对抗性噪声的传播，并以无监督的方式保护HoSNN的鲁棒性。我们还提出了理论分析，以阐明TA-LIF神经元的稳定性和收敛性，强调它们在输入多样性方面的卓越动态鲁棒性。

    Spiking neural networks (SNNs) offer promise for efficient and powerful neurally inspired computation. Common to other types of neural networks, however, SNNs face the severe issue of vulnerability to adversarial attacks. We present the first study that draws inspiration from neural homeostasis to develop a bio-inspired solution that counters the susceptibilities of SNNs to adversarial onslaughts. At the heart of our approach is a novel threshold-adapting leaky integrate-and-fire (TA-LIF) neuron model, which we adopt to construct the proposed adversarially robust homeostatic SNN (HoSNN). Distinct from traditional LIF models, our TA-LIF model incorporates a self-stabilizing dynamic thresholding mechanism, curtailing adversarial noise propagation and safeguarding the robustness of HoSNNs in an unsupervised manner. Theoretical analysis is presented to shed light on the stability and convergence properties of the TA-LIF neurons, underscoring their superior dynamic robustness under input di
    

