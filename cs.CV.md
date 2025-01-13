# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fractional Concepts in Neural Networks: Enhancing Activation and Loss Functions.](http://arxiv.org/abs/2310.11875) | 本文介绍了一种使用分数概念修改激活和损失函数的方法，通过调整神经元的激活函数，能够更好地适应输入数据并提高网络的性能。 |
| [^2] | [Fixating on Attention: Integrating Human Eye Tracking into Vision Transformers.](http://arxiv.org/abs/2308.13969) | 本研究展示了如何将人眼追踪集成到视觉Transformer模型中，提高在多种驾驶情况和数据集上的准确性。 |
| [^3] | [Discriminative Class Tokens for Text-to-Image Diffusion Models.](http://arxiv.org/abs/2303.17155) | 本文提出了一种基于判别性信息和自由文本的非侵入式微调技术，以实现多样性和高准确率的文本到图像生成模型。 |
| [^4] | [Lightweight Parameter Pruning for Energy-Efficient Deep Learning: A Binarized Gating Module Approach.](http://arxiv.org/abs/2302.10798) | 本论文提出了一种轻量化参数裁剪策略来实现节能深度学习，可以发现最佳的静态子网络，包含二值门控模块和新颖的损失函数，适用于各种神经网络，同时在已有的基准测试中取得了竞争性结果。 |
| [^5] | [Adversarial Detection by Approximation of Ensemble Boundary.](http://arxiv.org/abs/2211.10227) | 本论文提出了一种使用Walsh系数逼近决策边界的对抗攻击检测方法，通过观察清晰图像和对抗图像之间的Walsh系数逼近差异，实现了对对抗攻击的检测。 |
| [^6] | [Self-supervised video pretraining yields human-aligned visual representations.](http://arxiv.org/abs/2210.06433) | 本论文通过自监督训练的视频预训练方法VITO得到了具有人类感知特征的视觉表示，该方法在图像理解和视频理解任务上表现出更好的泛化性和鲁棒性。 |

# 详细

[^1]: 神经网络中的分数概念：增进激活和损失函数

    Fractional Concepts in Neural Networks: Enhancing Activation and Loss Functions. (arXiv:2310.11875v1 [cs.LG])

    [http://arxiv.org/abs/2310.11875](http://arxiv.org/abs/2310.11875)

    本文介绍了一种使用分数概念修改激活和损失函数的方法，通过调整神经元的激活函数，能够更好地适应输入数据并提高网络的性能。

    

    本文介绍了一种在神经网络中使用分数概念修改激活和损失函数的方法。该方法允许神经网络通过确定训练过程的分数导数阶数作为额外的超参数来定义和优化其激活函数。这将使得网络中的神经元能够调整其激活函数以更好地适应输入数据并减少输出错误，有可能提高网络的整体性能。

    The paper presents a method for using fractional concepts in a neural network to modify the activation and loss functions. The methodology allows the neural network to define and optimize its activation functions by determining the fractional derivative order of the training process as an additional hyperparameter. This will enable neurons in the network to adjust their activation functions to match input data better and reduce output errors, potentially improving the network's overall performance.
    
[^2]: 注重注意力：将人眼追踪集成到视觉Transformer模型中

    Fixating on Attention: Integrating Human Eye Tracking into Vision Transformers. (arXiv:2308.13969v1 [cs.CV])

    [http://arxiv.org/abs/2308.13969](http://arxiv.org/abs/2308.13969)

    本研究展示了如何将人眼追踪集成到视觉Transformer模型中，提高在多种驾驶情况和数据集上的准确性。

    

    现代基于Transformer的计算机视觉模型在多种视觉任务上表现出超越人类的能力。然而，一些关键任务，如医学图像解释和自动驾驶，仍然需要依赖人类判断。本研究展示了如何将人类视觉输入，特别是通过眼动仪收集到的注视点，集成到Transformer模型中，以提高在多种驾驶情况和数据集上的准确性。首先，我们在人类实验对象和Vision Transformer模型中观察到，注视区域在左右驾驶决策中的重要性。通过比较人类注视图和ViT注意力权重之间的相似性，我们揭示了单个头部和层之间的重叠动态。通过利用这种重叠动态，我们实现了对模型的修剪而不损失准确性。然后，我们将驾驶场景信息与注视数据相结合，采用“联合空间-注视”（JSF）的注意力设置。最后

    Modern transformer-based models designed for computer vision have outperformed humans across a spectrum of visual tasks. However, critical tasks, such as medical image interpretation or autonomous driving, still require reliance on human judgments. This work demonstrates how human visual input, specifically fixations collected from an eye-tracking device, can be integrated into transformer models to improve accuracy across multiple driving situations and datasets. First, we establish the significance of fixation regions in left-right driving decisions, as observed in both human subjects and a Vision Transformer (ViT). By comparing the similarity between human fixation maps and ViT attention weights, we reveal the dynamics of overlap across individual heads and layers. This overlap is exploited for model pruning without compromising accuracy. Thereafter, we incorporate information from the driving scene with fixation data, employing a "joint space-fixation" (JSF) attention setup. Lastly
    
[^3]: 基于判别性类标的文本图片扩散模型

    Discriminative Class Tokens for Text-to-Image Diffusion Models. (arXiv:2303.17155v1 [cs.CV])

    [http://arxiv.org/abs/2303.17155](http://arxiv.org/abs/2303.17155)

    本文提出了一种基于判别性信息和自由文本的非侵入式微调技术，以实现多样性和高准确率的文本到图像生成模型。

    

    最近文本到图像扩散模型的进展使得生成多样且高质量图片成为可能。然而，由于输入文本的歧义，生成的图片常常无法描绘出微妙的细节且易于出错。缓解这些问题的方法之一是在有类标注的数据集上训练扩散模型。这种方法的缺点在于：（i）与用于训练文本到图像模型的大规模爬取的文本-图像数据集相比，有类标注的数据集通常较小，因此生成的图片质量和多样性会严重受影响，或（ii）输入是硬编码的标签，而不是自由文本，这限制了对生成的图像的控制。在这项工作中，我们提出了一种非侵入式的微调技术，利用预训练分类器的判别性信号引导生成过程，既发挥了自由文本的表达潜力，又能够实现高准确率。

    Recent advances in text-to-image diffusion models have enabled the generation of diverse and high-quality images. However, generated images often fall short of depicting subtle details and are susceptible to errors due to ambiguity in the input text. One way of alleviating these issues is to train diffusion models on class-labeled datasets. This comes with a downside, doing so limits their expressive power: (i) supervised datasets are generally small compared to large-scale scraped text-image datasets on which text-to-image models are trained, and so the quality and diversity of generated images are severely affected, or (ii) the input is a hard-coded label, as opposed to free-form text, which limits the control over the generated images.  In this work, we propose a non-invasive fine-tuning technique that capitalizes on the expressive potential of free-form text while achieving high accuracy through discriminative signals from a pretrained classifier, which guides the generation. This 
    
[^4]: 轻量化参数裁剪以实现节能深度学习: 一种二值门控模块方法

    Lightweight Parameter Pruning for Energy-Efficient Deep Learning: A Binarized Gating Module Approach. (arXiv:2302.10798v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.10798](http://arxiv.org/abs/2302.10798)

    本论文提出了一种轻量化参数裁剪策略来实现节能深度学习，可以发现最佳的静态子网络，包含二值门控模块和新颖的损失函数，适用于各种神经网络，同时在已有的基准测试中取得了竞争性结果。

    

    随着神经网络越来越大且更加复杂，绿色AI已经引起了深度学习社区的关注。现有的解决方案通常采用对网络参数进行裁剪，以减少训练推断时的计算负荷。然而，裁剪方案通常会导致额外的开销，因为需要进行迭代训练和微调或重复计算动态裁剪图。我们提出了一种新的参数裁剪策略，以学习轻量级子网络，既能最小化能耗，又能在给定的下游任务上与完全参数化的网络保持相似的性能。我们的裁剪方案以绿色为导向，因为它仅需要动态裁剪方法进行一次训练来发现最佳的静态子网络。该方案由一个二进制门控模块和一个新颖的损失函数组成，以发现具有用户定义稀疏度的子网络。我们的方法可以对卷积神经网络（CNN）和循环神经网络（RNN）等通用神经网络进行裁剪和转换，大大减少了计算复杂度和能量消耗，同时在已有的基准测试上取得了竞争性的结果。

    The subject of green AI has been gaining attention within the deep learning community given the recent trend of ever larger and more complex neural network models. Existing solutions for reducing the computational load of training at inference time usually involve pruning the network parameters. Pruning schemes often create extra overhead either by iterative training and fine-tuning for static pruning or repeated computation of a dynamic pruning graph. We propose a new parameter pruning strategy for learning a lighter-weight sub-network that minimizes the energy cost while maintaining comparable performance to the fully parameterised network on given downstream tasks. Our proposed pruning scheme is green-oriented, as it only requires a one-off training to discover the optimal static sub-networks by dynamic pruning methods. The pruning scheme consists of a binary gating module and a novel loss function to uncover sub-networks with user-defined sparsity. Our method enables pruning and tr
    
[^5]: 使用集成边界逼近的对抗检测方法

    Adversarial Detection by Approximation of Ensemble Boundary. (arXiv:2211.10227v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.10227](http://arxiv.org/abs/2211.10227)

    本论文提出了一种使用Walsh系数逼近决策边界的对抗攻击检测方法，通过观察清晰图像和对抗图像之间的Walsh系数逼近差异，实现了对对抗攻击的检测。

    

    本论文提出了一种新的对抗攻击检测方法，针对解决两类模式识别问题的深度神经网络（DNN）集成。该集成使用Walsh系数进行组合，能够逼近布尔函数并控制集成决策边界的复杂性。本文的假设是高曲率的决策边界允许找到对抗扰动，但会改变决策边界的曲率，而与清晰图像相比，使用Walsh系数对其进行逼近的方式也有所不同。通过观察清晰图像和对抗图像之间的Walsh系数逼近差异，实验证明了攻击的可迁移性可用于检测。此外，逼近决策边界可能有助于理解DNN的学习和可迁移性特性。尽管本文的实验使用图像，所提出的方法可以用于建模两类模式识别问题的集成边界逼近。

    A new method of detecting adversarial attacks is proposed for an ensemble of Deep Neural Networks (DNNs) solving two-class pattern recognition problems. The ensemble is combined using Walsh coefficients which are capable of approximating Boolean functions and thereby controlling the complexity of the ensemble decision boundary. The hypothesis in this paper is that decision boundaries with high curvature allow adversarial perturbations to be found, but change the curvature of the decision boundary, which is then approximated in a different way by Walsh coefficients compared to the clean images. By observing the difference in Walsh coefficient approximation between clean and adversarial images, it is shown experimentally that transferability of attack may be used for detection. Furthermore, approximating the decision boundary may aid in understanding the learning and transferability properties of DNNs. While the experiments here use images, the proposed approach of modelling two-class en
    
[^6]: 自监督训练的视频预训练产生与人类对齐的视觉表示

    Self-supervised video pretraining yields human-aligned visual representations. (arXiv:2210.06433v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2210.06433](http://arxiv.org/abs/2210.06433)

    本论文通过自监督训练的视频预训练方法VITO得到了具有人类感知特征的视觉表示，该方法在图像理解和视频理解任务上表现出更好的泛化性和鲁棒性。

    

    人类通过观察对象和场景随时间演变的方式学习到了强大的表示。然而，在不需要明确的时间理解的特定任务之外，静态图像预训练仍然是学习视觉基础模型的主流范式。我们对这种不匹配提出了质疑，并且问是否视频预训练可以产生具有人类感知特征的视觉表示：在各种任务中的泛化性、对扰动的鲁棒性和与人类判断的一致性。为此，我们提出了一种用于筛选视频的新颖程序，并开发了一个对比性框架，从其中的复杂转换中学习。这种从视频中提炼知识的简单范式被称为VITO，它产生的一般表示在图像理解任务上远远优于先前的视频预训练方法，并且在视频理解任务上优于图像预训练方法。此外，VITO表示对自然和合成形变的鲁棒性显著提高。

    Humans learn powerful representations of objects and scenes by observing how they evolve over time. Yet, outside of specific tasks that require explicit temporal understanding, static image pretraining remains the dominant paradigm for learning visual foundation models. We question this mismatch, and ask whether video pretraining can yield visual representations that bear the hallmarks of human perception: generalisation across tasks, robustness to perturbations, and consistency with human judgements. To that end we propose a novel procedure for curating videos, and develop a contrastive framework which learns from the complex transformations therein. This simple paradigm for distilling knowledge from videos, called VITO, yields general representations that far outperform prior video pretraining methods on image understanding tasks, and image pretraining methods on video understanding tasks. Moreover, VITO representations are significantly more robust to natural and synthetic deformati
    

