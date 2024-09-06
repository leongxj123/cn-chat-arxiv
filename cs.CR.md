# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [How to Train your Antivirus: RL-based Hardening through the Problem-Space](https://arxiv.org/abs/2402.19027) | 引入了一种基于强化学习的方法，可在问题空间内构建对抗样本，对抗防病毒软件中的恶意软件攻击。 |
| [^2] | [AICAttack: Adversarial Image Captioning Attack with Attention-Based Optimization](https://arxiv.org/abs/2402.11940) | 提出了一种新的对抗攻击策略AICAttack，旨在通过微小的图像扰动来攻击图像字幕模型，在黑盒攻击情景下具有良好的效果。 |
| [^3] | [TSFool: Crafting Highly-imperceptible Adversarial Time Series through Multi-objective Black-box Attack to Fool RNN Classifiers.](http://arxiv.org/abs/2209.06388) | 本文提出了一种名为TSFool的黑盒方法, 可以有效地生成针对RNN分类器的高度难以察觉的对抗性时间序列，在考虑对抗样本难以察觉性的情况下，将对抗性攻击改进为多目标优化问题来增强扰动的质量。 |

# 详细

[^1]: 如何训练您的防病毒软件：基于强化学习的问题空间加固

    How to Train your Antivirus: RL-based Hardening through the Problem-Space

    [https://arxiv.org/abs/2402.19027](https://arxiv.org/abs/2402.19027)

    引入了一种基于强化学习的方法，可在问题空间内构建对抗样本，对抗防病毒软件中的恶意软件攻击。

    

    本文探讨了一种特定的机器学习架构，用于加固一家著名商业防病毒公司流程中的机器学习防御技术，以对抗恶意软件。我们引入了一种新颖的强化学习方法，用于构建对抗样本，这是对抗逃避攻击的模型训练的重要组成部分。

    arXiv:2402.19027v1 Announce Type: cross  Abstract: ML-based malware detection on dynamic analysis reports is vulnerable to both evasion and spurious correlations. In this work, we investigate a specific ML architecture employed in the pipeline of a widely-known commercial antivirus company, with the goal to harden it against adversarial malware. Adversarial training, the sole defensive technique that can confer empirical robustness, is not applicable out of the box in this domain, for the principal reason that gradient-based perturbations rarely map back to feasible problem-space programs. We introduce a novel Reinforcement Learning approach for constructing adversarial examples, a constituent part of adversarially training a model against evasion. Our approach comes with multiple advantages. It performs modifications that are feasible in the problem-space, and only those; thus it circumvents the inverse mapping problem. It also makes possible to provide theoretical guarantees on the r
    
[^2]: AICAttack：基于注意力优化的对抗性图像字幕攻击

    AICAttack: Adversarial Image Captioning Attack with Attention-Based Optimization

    [https://arxiv.org/abs/2402.11940](https://arxiv.org/abs/2402.11940)

    提出了一种新的对抗攻击策略AICAttack，旨在通过微小的图像扰动来攻击图像字幕模型，在黑盒攻击情景下具有良好的效果。

    

    最近深度学习研究取得了在计算机视觉（CV）和自然语言处理（NLP）等许多任务上显著的成就。CV和NLP交叉点上的图像字幕问题中，相关模型对抗攻击的稳健性尚未得到充分研究。本文提出了一种新颖的对抗攻击策略，称为AICAttack（基于注意力的图像字幕攻击），旨在通过对图像进行微小扰动来攻击图像字幕模型。在黑盒攻击环境中运行，我们的算法不需要访问目标模型的架构、参数或梯度信息。我们引入了基于注意力的候选选择机制，可识别最佳像素进行攻击，然后采用差分进化（DE）来扰乱像素的RGB值。通过对基准上的广泛实验，我们证明了AICAttack的有效性。

    arXiv:2402.11940v1 Announce Type: cross  Abstract: Recent advances in deep learning research have shown remarkable achievements across many tasks in computer vision (CV) and natural language processing (NLP). At the intersection of CV and NLP is the problem of image captioning, where the related models' robustness against adversarial attacks has not been well studied. In this paper, we present a novel adversarial attack strategy, which we call AICAttack (Attention-based Image Captioning Attack), designed to attack image captioning models through subtle perturbations on images. Operating within a black-box attack scenario, our algorithm requires no access to the target model's architecture, parameters, or gradient information. We introduce an attention-based candidate selection mechanism that identifies the optimal pixels to attack, followed by Differential Evolution (DE) for perturbing pixels' RGB values. We demonstrate AICAttack's effectiveness through extensive experiments on benchma
    
[^3]: TSFool: 通过多目标黑盒攻击方法生成高度难以察觉的对循环神经网络分类器的对抗性时间序列

    TSFool: Crafting Highly-imperceptible Adversarial Time Series through Multi-objective Black-box Attack to Fool RNN Classifiers. (arXiv:2209.06388v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.06388](http://arxiv.org/abs/2209.06388)

    本文提出了一种名为TSFool的黑盒方法, 可以有效地生成针对RNN分类器的高度难以察觉的对抗性时间序列，在考虑对抗样本难以察觉性的情况下，将对抗性攻击改进为多目标优化问题来增强扰动的质量。

    

    神经网络分类器很容易受到对抗性攻击。现有的梯度攻击方法在前馈神经网络和图像识别任务中取得了最先进的性能，但它们在循环神经网络模型下的时间序列分类中表现不佳。这是因为RNN的循环结构阻止了直接的模型差分，而时间序列数据对扰动的视觉敏感性挑战了对抗性攻击的传统局部优化目标。本文提出了一种名为TSFool的黑盒方法，用于有效地生成针对RNN分类器的高度难以察觉的对抗性时间序列。我们提出了一种新的全局优化目标，称为Camouflage Coefficient，从类分布的角度考虑对抗样本的难以察觉性，并相应地将对抗性攻击改进为多目标优化问题，以增强扰动的质量。为了摆脱不同模型间的转移性，设计了一个特定于模型的回避规则。在人造数据和实际数据集上的实验结果表明，TSFool可以生成高难度攻击同时保持对抗样本的不易被检测性，并有很高的转移性。

    Neural network (NN) classifiers are vulnerable to adversarial attacks. Although the existing gradient-based attacks achieve state-of-the-art performance in feed-forward NNs and image recognition tasks, they do not perform as well on time series classification with recurrent neural network (RNN) models. This is because the cyclical structure of RNN prevents direct model differentiation and the visual sensitivity of time series data to perturbations challenges the traditional local optimization objective of the adversarial attack. In this paper, a black-box method called TSFool is proposed to efficiently craft highly-imperceptible adversarial time series for RNN classifiers. We propose a novel global optimization objective named Camouflage Coefficient to consider the imperceptibility of adversarial samples from the perspective of class distribution, and accordingly refine the adversarial attack as a multi-objective optimization problem to enhance the perturbation quality. To get rid of t
    

