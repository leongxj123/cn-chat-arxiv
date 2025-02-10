# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ADAPT to Robustify Prompt Tuning Vision Transformers](https://arxiv.org/abs/2403.13196) | 本文提出了ADAPT框架，用于在prompt调优范式中进行自适应对抗训练，增强视觉Transformer在下游任务中的稳健性。 |
| [^2] | [A Bayesian Approach to OOD Robustness in Image Classification](https://arxiv.org/abs/2403.07277) | 本文提出了一种基于贝叶斯方法的图像分类中OOD鲁棒性解决方案，利用扩展的组合神经网络和von Mises-Fisher核来处理真实世界的OOD问题。 |

# 详细

[^1]: 使Prompt调优视觉Transformer更为健壮的ADAPT

    ADAPT to Robustify Prompt Tuning Vision Transformers

    [https://arxiv.org/abs/2403.13196](https://arxiv.org/abs/2403.13196)

    本文提出了ADAPT框架，用于在prompt调优范式中进行自适应对抗训练，增强视觉Transformer在下游任务中的稳健性。

    

    深度模型的性能，包括视觉Transformer，已知容易受到对抗性攻击的影响。许多现有对抗性防御方法，如对抗性训练，依赖于对整个模型进行全面微调以增加模型的稳健性。这些防御方法需要为每个任务存储整个模型的副本，而模型可能包含数十亿个参数。与此同时，参数高效的prompt调优被用来适应大型基于Transformer的模型到下游任务，无需保存大型副本。本文从稳健性的角度研究了对视觉Transformer进行下游任务的参数高效prompt调优。我们发现，之前的对抗性防御方法在应用到prompt调优范式时，存在梯度模糊并容易受到自适应攻击的影响。我们引入了ADAPT，一种在prompt调优范式中执行自适应对抗训练的新框架。

    arXiv:2403.13196v1 Announce Type: new  Abstract: The performance of deep models, including Vision Transformers, is known to be vulnerable to adversarial attacks. Many existing defenses against these attacks, such as adversarial training, rely on full-model fine-tuning to induce robustness in the models. These defenses require storing a copy of the entire model, that can have billions of parameters, for each task. At the same time, parameter-efficient prompt tuning is used to adapt large transformer-based models to downstream tasks without the need to save large copies. In this paper, we examine parameter-efficient prompt tuning of Vision Transformers for downstream tasks under the lens of robustness. We show that previous adversarial defense methods, when applied to the prompt tuning paradigm, suffer from gradient obfuscation and are vulnerable to adaptive attacks. We introduce ADAPT, a novel framework for performing adaptive adversarial training in the prompt tuning paradigm. Our meth
    
[^2]: 基于贝叶斯方法的图像分类中OOD鲁棒性解决方案

    A Bayesian Approach to OOD Robustness in Image Classification

    [https://arxiv.org/abs/2403.07277](https://arxiv.org/abs/2403.07277)

    本文提出了一种基于贝叶斯方法的图像分类中OOD鲁棒性解决方案，利用扩展的组合神经网络和von Mises-Fisher核来处理真实世界的OOD问题。

    

    计算机视觉中一个重要且未解决的问题是确保算法对图像领域的变化具有鲁棒性。我们在目标领域中处理此问题的情况下，但没有注释的图像。在面临真实世界的域之外（OOD）干扰和遮挡的OOD-CV基准挑战的激励下，我们引入了一种新颖的贝叶斯方法来实现物体分类的OOD鲁棒性。我们的工作扩展了已被证明在遮挡情况下具有鲁棒性但在OOD数据测试时严重降级的组合神经网络（CompNets）。我们利用了CompNets包含的在von Mises-Fisher（vMF）核表示的特征向量上定义的生成头，这些核大致对应于对象部分，并且可以在无监督的情况下学习。我们观察到不同域之间的某些vMF核是相似的，而另一些则不是。这使我们能够学习一个transiti

    arXiv:2403.07277v1 Announce Type: cross  Abstract: An important and unsolved problem in computer vision is to ensure that the algorithms are robust to changes in image domains. We address this problem in the scenario where we have access to images from the target domains but no annotations. Motivated by the challenges of the OOD-CV benchmark where we encounter real world Out-of-Domain (OOD) nuisances and occlusion, we introduce a novel Bayesian approach to OOD robustness for object classification. Our work extends Compositional Neural Networks (CompNets), which have been shown to be robust to occlusion but degrade badly when tested on OOD data. We exploit the fact that CompNets contain a generative head defined over feature vectors represented by von Mises-Fisher (vMF) kernels, which correspond roughly to object parts, and can be learned without supervision. We obverse that some vMF kernels are similar between different domains, while others are not. This enables us to learn a transiti
    

