# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Curious Case of Remarkable Resilience to Gradient Attacks via Fully Convolutional and Differentiable Front End with a Skip Connection](https://arxiv.org/abs/2402.17018) | 通过在神经模型中引入不同iable和完全卷积的前端模型，并结合跳跃连接，成功实现对梯度攻击的显著韧性，并通过将模型组合成随机集合，有效对抗黑盒攻击。 |
| [^2] | [Distilling Inductive Bias: Knowledge Distillation Beyond Model Compression.](http://arxiv.org/abs/2310.00369) | 本文提出了一种超越模型压缩的知识蒸馏方法，通过从轻量级教师模型中提取归纳偏差，使Vision Transformers (ViTs) 的应用成为可能。这种方法包括使用一组不同架构的教师模型来指导学生Transformer，从而有效提高学生的性能。 |

# 详细

[^1]: 通过完全卷积和可微的前端与跳跃连接对梯度攻击表现出显著韧性的耐人寻味案例

    A Curious Case of Remarkable Resilience to Gradient Attacks via Fully Convolutional and Differentiable Front End with a Skip Connection

    [https://arxiv.org/abs/2402.17018](https://arxiv.org/abs/2402.17018)

    通过在神经模型中引入不同iable和完全卷积的前端模型，并结合跳跃连接，成功实现对梯度攻击的显著韧性，并通过将模型组合成随机集合，有效对抗黑盒攻击。

    

    我们测试了通过在一个冻结的分类器之前增加一个可微且完全卷积的模型，并具有跳跃连接的前端增强神经模型。通过使用较小的学习率进行大约一个epoch的训练，我们获得了一些模型，这些模型在保持骨干分类器准确性的同时，对包括AutoAttack软件包中的APGD和FAB-T攻击在内的梯度攻击具有异常的抵抗力，这归因于梯度掩盖。梯度掩盖现象并不新鲜，但对于这些没有梯度破坏部分（如JPEG压缩或预计导致梯度减小的部分）的完全可微模型来说，掩盖的程度相当显著。尽管黑盒攻击对梯度掩盖可能部分有效，但通过将模型组合成随机集合，可以轻松击败它们。我们估计这样的集合在CIFAR10和CIF等上实现了几乎SOTA级别的AutoAttack准确性。

    arXiv:2402.17018v1 Announce Type: cross  Abstract: We tested front-end enhanced neural models where a frozen classifier was prepended by a differentiable and fully convolutional model with a skip connection. By training them using a small learning rate for about one epoch, we obtained models that retained the accuracy of the backbone classifier while being unusually resistant to gradient attacks including APGD and FAB-T attacks from the AutoAttack package, which we attributed to gradient masking. The gradient masking phenomenon is not new, but the degree of masking was quite remarkable for fully differentiable models that did not have gradient-shattering components such as JPEG compression or components that are expected to cause diminishing gradients.   Though black box attacks can be partially effective against gradient masking, they are easily defeated by combining models into randomized ensembles. We estimate that such ensembles achieve near-SOTA AutoAttack accuracy on CIFAR10, CIF
    
[^2]: 提炼归纳偏差：超越模型压缩的知识蒸馏

    Distilling Inductive Bias: Knowledge Distillation Beyond Model Compression. (arXiv:2310.00369v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2310.00369](http://arxiv.org/abs/2310.00369)

    本文提出了一种超越模型压缩的知识蒸馏方法，通过从轻量级教师模型中提取归纳偏差，使Vision Transformers (ViTs) 的应用成为可能。这种方法包括使用一组不同架构的教师模型来指导学生Transformer，从而有效提高学生的性能。

    

    随着计算机视觉的快速发展，Vision Transformers (ViTs) 提供了在视觉和文本领域中实现统一信息处理的诱人前景。但是由于ViTs缺乏固有的归纳偏差，它们需要大量的训练数据。为了使它们的应用实际可行，我们引入了一种创新的基于集成的蒸馏方法，从轻量级的教师模型中提取归纳偏差。以前的系统仅依靠基于卷积的教学方法。然而，这种方法将一组具有不同架构倾向的轻量级教师模型（例如卷积和非线性卷积）同时用于指导学生Transformer。由于这些独特的归纳偏差，教师模型可以从各种存储数据集中获得广泛的知识，从而提高学生的性能。我们提出的框架还涉及预先计算和存储logits，从根本上实现了非归一化的状态匹配。

    With the rapid development of computer vision, Vision Transformers (ViTs) offer the tantalizing prospect of unified information processing across visual and textual domains. But due to the lack of inherent inductive biases in ViTs, they require enormous amount of data for training. To make their applications practical, we introduce an innovative ensemble-based distillation approach distilling inductive bias from complementary lightweight teacher models. Prior systems relied solely on convolution-based teaching. However, this method incorporates an ensemble of light teachers with different architectural tendencies, such as convolution and involution, to instruct the student transformer jointly. Because of these unique inductive biases, instructors can accumulate a wide range of knowledge, even from readily identifiable stored datasets, which leads to enhanced student performance. Our proposed framework also involves precomputing and storing logits in advance, essentially the unnormalize
    

