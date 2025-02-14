# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Cognitive Evaluation Benchmark of Image Reasoning and Description for Large Vision Language Models](https://arxiv.org/abs/2402.18409) | 提出了一个新颖的评估基准，用于评估大型视觉语言模型的认知能力，发现LVLMs与人类之间存在较大的认知能力差距。 |
| [^2] | [Opening Cabinets and Drawers in the Real World using a Commodity Mobile Manipulator](https://arxiv.org/abs/2402.17767) | 实现了一个端到端系统，使商品移动操作器成功在以前未见的真实世界环境中打开橱柜和抽屉，感知误差是主要挑战。 |
| [^3] | [For Better or For Worse? Learning Minimum Variance Features With Label Augmentation](https://arxiv.org/abs/2402.06855) | 本研究分析了标签增强方法中标签增强的作用。研究证明，在线性可分数据上使用标签增强训练的线性模型只能学习到最小方差特征，而标准训练可以学习到更高方差特征。此外，标签平滑和Mixup对于训练数据的对抗扰动可能不太鲁棒。 |
| [^4] | [Vision-LLMs Can Fool Themselves with Self-Generated Typographic Attacks](https://arxiv.org/abs/2402.00626) | 这项研究深入研究了大规模视觉语言模型（LVLM）对于自动生成的排版攻击的易受攻击性，并引入了一种新的、更有效的自动生成的排版攻击方法，为此设计了一个独特的测试基准。通过使用该基准，研究发现排版攻击对LVLM构成了重大威胁。 |
| [^5] | [Multi-level Asymmetric Contrastive Learning for Medical Image Segmentation Pre-training.](http://arxiv.org/abs/2309.11876) | 本论文提出了一种针对医学图像分割的自我监督预训练方法，通过多级非对称对比学习的框架，在编码器和解码器同时进行预训练，提供更好的分割模型初始化。 |
| [^6] | [MonoDETR: Depth-guided Transformer for Monocular 3D Object Detection.](http://arxiv.org/abs/2203.13310) | 本文介绍了一种名为MonoDETR的深度引导Transformer框架，用于单目3D目标检测。相比于传统的方法，MonoDETR通过引入深度信息来指导整个检测过程，提高了对场景的理解和目标的准确性。 |

# 详细

[^1]: 一个针对大型视觉语言模型图像推理和描述的认知评估基准

    A Cognitive Evaluation Benchmark of Image Reasoning and Description for Large Vision Language Models

    [https://arxiv.org/abs/2402.18409](https://arxiv.org/abs/2402.18409)

    提出了一个新颖的评估基准，用于评估大型视觉语言模型的认知能力，发现LVLMs与人类之间存在较大的认知能力差距。

    

    尽管大型视觉语言模型(LVLMs)近年来取得了成功，但它们很少受到全面的认知能力测试。受到人类认知测试中广泛使用的“偷饼干”任务的启发，我们提出了一个新颖的评估基准，利用具有丰富语义的图像评估LVLMs的高级认知能力。它定义了八种推理能力，并包括图像描述任务和视觉问答任务。我们对知名LVLMs进行的评估表明，在LVLMs和人类之间仍存在较大的认知能力差距。

    arXiv:2402.18409v1 Announce Type: new  Abstract: Large Vision Language Models (LVLMs), despite their recent success, are hardly comprehensively tested for their cognitive abilities. Inspired by the prevalent use of the "Cookie Theft" task in human cognition test, we propose a novel evaluation benchmark to evaluate high-level cognitive ability of LVLMs using images with rich semantics. It defines eight reasoning capabilities and consists of an image description task and a visual question answering task. Our evaluation on well-known LVLMs shows that there is still a large gap in cognitive ability between LVLMs and humans.
    
[^2]: 在现实世界中使用商品移动操作器打开橱柜和抽屉

    Opening Cabinets and Drawers in the Real World using a Commodity Mobile Manipulator

    [https://arxiv.org/abs/2402.17767](https://arxiv.org/abs/2402.17767)

    实现了一个端到端系统，使商品移动操作器成功在以前未见的真实世界环境中打开橱柜和抽屉，感知误差是主要挑战。

    

    在这项工作中，我们构建了一个端到端系统，使商品移动操作器（Stretch RE2）能够在多样的以前未见的真实世界环境中拉开橱柜和抽屉。我们在31个不同的物体和13个不同真实世界环境中进行了4天的实际测试。我们的系统在零击打下，对在未知环境中新颖的橱柜和抽屉的打开率达到61%。对失败模式的分析表明，感知误差是我们系统面临的最重要挑战。

    arXiv:2402.17767v1 Announce Type: cross  Abstract: Pulling open cabinets and drawers presents many difficult technical challenges in perception (inferring articulation parameters for objects from onboard sensors), planning (producing motion plans that conform to tight task constraints), and control (making and maintaining contact while applying forces on the environment). In this work, we build an end-to-end system that enables a commodity mobile manipulator (Stretch RE2) to pull open cabinets and drawers in diverse previously unseen real world environments. We conduct 4 days of real world testing of this system spanning 31 different objects from across 13 different real world environments. Our system achieves a success rate of 61% on opening novel cabinets and drawers in unseen environments zero-shot. An analysis of the failure modes suggests that errors in perception are the most significant challenge for our system. We will open source code and models for others to replicate and bui
    
[^3]: 更好还是更差？通过标签增强学习最小方差特征

    For Better or For Worse? Learning Minimum Variance Features With Label Augmentation

    [https://arxiv.org/abs/2402.06855](https://arxiv.org/abs/2402.06855)

    本研究分析了标签增强方法中标签增强的作用。研究证明，在线性可分数据上使用标签增强训练的线性模型只能学习到最小方差特征，而标准训练可以学习到更高方差特征。此外，标签平滑和Mixup对于训练数据的对抗扰动可能不太鲁棒。

    

    在过去的十年中，数据增强对于成功地训练深度学习模型在分类任务上发挥了关键作用。数据增强技术中的一个重要子类-包括标签平滑和Mixup-涉及在模型训练过程中修改输入数据和输入标签。在这项工作中，我们分析了此类方法中标签增强的作用。我们证明了在线性可分数据上使用标签增强训练的线性模型只能学习到最小方差特征，而标准训练（包括权重衰减）可以学习到更高方差特征。我们的结果的一个重要后果是消极的：与标准训练相比，标签平滑和Mixup对于训练数据的对抗扰动可能不太鲁棒。我们通过对合成数据和图像分类基准的一系列实验证明了我们的理论与实践的一致性。

    Data augmentation has been pivotal in successfully training deep learning models on classification tasks over the past decade. An important subclass of data augmentation techniques - which includes both label smoothing and Mixup - involves modifying not only the input data but also the input label during model training. In this work, we analyze the role played by the label augmentation aspect of such methods. We prove that linear models on linearly separable data trained with label augmentation learn only the minimum variance features in the data, while standard training (which includes weight decay) can learn higher variance features. An important consequence of our results is negative: label smoothing and Mixup can be less robust to adversarial perturbations of the training data when compared to standard training. We verify that our theory reflects practice via a range of experiments on synthetic data and image classification benchmarks.
    
[^4]: Vision-LLMs通过自动生成的排版攻击可以自欺欺人

    Vision-LLMs Can Fool Themselves with Self-Generated Typographic Attacks

    [https://arxiv.org/abs/2402.00626](https://arxiv.org/abs/2402.00626)

    这项研究深入研究了大规模视觉语言模型（LVLM）对于自动生成的排版攻击的易受攻击性，并引入了一种新的、更有效的自动生成的排版攻击方法，为此设计了一个独特的测试基准。通过使用该基准，研究发现排版攻击对LVLM构成了重大威胁。

    

    最近，在大规模视觉语言模型（LVLM）方面取得了重大进展；这是一种利用大型预训练语言模型的全新类别的视觉语言模型。然而，LVLM对于涉及将误导性文本叠加到图像上的从排版攻击的容易受攻击性却没有研究。此外，先前的排版攻击依赖于从预定义类别集合中随机选择一个误导性类别。然而，随机选择的类别可能不是最有效的攻击类别。为了解决这些问题，我们首先引入了一种独特设计的新颖基准来测试LVLM对排版攻击的容易受攻击性。此外，我们介绍了一种新而更有效的排版攻击：自动生成的排版攻击。实际上，我们的方法通过简单地提示GPT-4V等模型利用其强大的语言能力推荐一种排版攻击来为给定的图像生成攻击。使用我们的新颖基准，我们发现排版攻击对LVLM构成了重大威胁。

    Recently, significant progress has been made on Large Vision-Language Models (LVLMs); a new class of VL models that make use of large pre-trained language models. Yet, their vulnerability to Typographic attacks, which involve superimposing misleading text onto an image remain unstudied. Furthermore, prior work typographic attacks rely on sampling a random misleading class from a predefined set of classes. However, the random chosen class might not be the most effective attack. To address these issues, we first introduce a novel benchmark uniquely designed to test LVLMs vulnerability to typographic attacks. Furthermore, we introduce a new and more effective typographic attack: Self-Generated typographic attacks. Indeed, our method, given an image, make use of the strong language capabilities of models like GPT-4V by simply prompting them to recommend a typographic attack. Using our novel benchmark, we uncover that typographic attacks represent a significant threat against LVLM(s). Furth
    
[^5]: 多级非对称对比学习在医学图像分割预训练中的应用

    Multi-level Asymmetric Contrastive Learning for Medical Image Segmentation Pre-training. (arXiv:2309.11876v1 [cs.CV])

    [http://arxiv.org/abs/2309.11876](http://arxiv.org/abs/2309.11876)

    本论文提出了一种针对医学图像分割的自我监督预训练方法，通过多级非对称对比学习的框架，在编码器和解码器同时进行预训练，提供更好的分割模型初始化。

    

    对比学习是一种从无标签数据中学习图像级表示的强大技术，为解决大规模预训练和有限标注数据之间的困境提供了一种有前途的方法。然而，大多数现有的对比学习策略主要针对自然图像的下游任务设计，因此当直接应用于医学图像（其下游任务通常是分割）时，它们往往是次优的甚至不如从头开始训练。在这项工作中，我们提出了一种名为JCL的新型非对称对比学习框架，用于医学图像分割的自我监督预训练。具体来说，（1）我们提出了一种新颖的非对称对比学习策略，同时在一阶段内对编码器和解码器进行预训练，以提供更好的分割模型初始化。 （2）我们设计了一个多级对比损失，用于考虑特征级别、图像级别和像素级别投影的对应关系。

    Contrastive learning, which is a powerful technique for learning image-level representations from unlabeled data, leads a promising direction to dealing with the dilemma between large-scale pre-training and limited labeled data. However, most existing contrastive learning strategies are designed mainly for downstream tasks of natural images, therefore they are sub-optimal and even worse than learning from scratch when directly applied to medical images whose downstream tasks are usually segmentation. In this work, we propose a novel asymmetric contrastive learning framework named JCL for medical image segmentation with self-supervised pre-training. Specifically, (1) A novel asymmetric contrastive learning strategy is proposed to pre-train both encoder and decoder simultaneously in one-stage to provide better initialization for segmentation models. (2) A multi-level contrastive loss is designed to take the correspondence among feature-level, image-level and pixel-level projections, resp
    
[^6]: MonoDETR：深度引导的单目3D目标检测的Transformer

    MonoDETR: Depth-guided Transformer for Monocular 3D Object Detection. (arXiv:2203.13310v4 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2203.13310](http://arxiv.org/abs/2203.13310)

    本文介绍了一种名为MonoDETR的深度引导Transformer框架，用于单目3D目标检测。相比于传统的方法，MonoDETR通过引入深度信息来指导整个检测过程，提高了对场景的理解和目标的准确性。

    

    单目三维目标检测一直是自动驾驶中一项具有挑战性的任务。大多数现有方法是根据传统的二维检测器首先定位目标中心，然后通过邻近特征预测三维属性。然而，仅仅使用局部视觉特征是不足以理解场景级别的三维空间结构并忽略了远距离的目标深度关系。在本文中，我们引入了第一个采用深度引导Transformer的单目检测框架，称为MonoDETR。我们将基本的Transformer进行了修改，使其具有深度感知，并通过上下文深度线索来指导整个检测过程。具体而言，在捕捉物体外观的视觉编码器的同时，我们引入了预测前景深度图，并专门设计了一个深度编码器来提取非局部深度嵌入。然后，我们将三维目标候选物形式化为可学习的查询，并提出了一个深度引导的解码器来进行目标-场景深度交互。通过这种方式，每个目标都可以得到更全面的深度感知和更准确的三维检测结果。

    Monocular 3D object detection has long been a challenging task in autonomous driving. Most existing methods follow conventional 2D detectors to first localize object centers, and then predict 3D attributes by neighboring features. However, only using local visual features is insufficient to understand the scene-level 3D spatial structures and ignores the long-range inter-object depth relations. In this paper, we introduce the first DETR framework for Monocular DEtection with a depth-guided TRansformer, named MonoDETR. We modify the vanilla transformer to be depth-aware and guide the whole detection process by contextual depth cues. Specifically, concurrent to the visual encoder that captures object appearances, we introduce to predict a foreground depth map, and specialize a depth encoder to extract non-local depth embeddings. Then, we formulate 3D object candidates as learnable queries and propose a depth-guided decoder to conduct object-scene depth interactions. In this way, each obj
    

