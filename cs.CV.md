# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CLEVRER-Humans: Describing Physical and Causal Events the Human Way.](http://arxiv.org/abs/2310.03635) | CLEVRER-Humans是一个用于因果判断的视频推理数据集，通过人工标注来解决合成事件和合成语言描述的缺乏多样性问题，并通过迭代事件填空和神经语言生成模型提高数据收集效率。 |
| [^2] | [Distilling Inductive Bias: Knowledge Distillation Beyond Model Compression.](http://arxiv.org/abs/2310.00369) | 本文提出了一种超越模型压缩的知识蒸馏方法，通过从轻量级教师模型中提取归纳偏差，使Vision Transformers (ViTs) 的应用成为可能。这种方法包括使用一组不同架构的教师模型来指导学生Transformer，从而有效提高学生的性能。 |
| [^3] | [Multiple Different Explanations for Image Classifiers.](http://arxiv.org/abs/2309.14309) | 这篇论文介绍了一种算法和工具，可以为图像分类器的输出计算多个解释，从而提高对分类器行为的洞察力。 |

# 详细

[^1]: CLEVRER-Humans: 用人类的方式描述物理和因果事件

    CLEVRER-Humans: Describing Physical and Causal Events the Human Way. (arXiv:2310.03635v1 [cs.AI])

    [http://arxiv.org/abs/2310.03635](http://arxiv.org/abs/2310.03635)

    CLEVRER-Humans是一个用于因果判断的视频推理数据集，通过人工标注来解决合成事件和合成语言描述的缺乏多样性问题，并通过迭代事件填空和神经语言生成模型提高数据收集效率。

    

    构建能够推理物理事件及其因果关系的机器对于与物理世界进行灵活互动非常重要。然而，现有的大多数物理和因果推理基准都仅基于合成事件和合成自然语言描述的因果关系。这种设计存在两个问题：一是事件类型和自然语言描述缺乏多样性；二是基于手动定义的启发式规则的因果关系与人类判断不一致。为了解决这两个问题，我们提出了CLEVRER-Humans基准，这是一个用人工标注的视频推理数据集，用于对物理事件的因果判断。我们采用了两种技术来提高数据收集效率：首先，一种新颖的迭代事件填空任务，以 eliciting 视频中事件的新表示方式，我们称之为因果事件图 (CEGs)；其次，一种基于神经语言生成模型的数据增强技术。

    Building machines that can reason about physical events and their causal relationships is crucial for flexible interaction with the physical world. However, most existing physical and causal reasoning benchmarks are exclusively based on synthetically generated events and synthetic natural language descriptions of causal relationships. This design brings up two issues. First, there is a lack of diversity in both event types and natural language descriptions; second, causal relationships based on manually-defined heuristics are different from human judgments. To address both shortcomings, we present the CLEVRER-Humans benchmark, a video reasoning dataset for causal judgment of physical events with human labels. We employ two techniques to improve data collection efficiency: first, a novel iterative event cloze task to elicit a new representation of events in videos, which we term Causal Event Graphs (CEGs); second, a data augmentation technique based on neural language generative models.
    
[^2]: 提炼归纳偏差：超越模型压缩的知识蒸馏

    Distilling Inductive Bias: Knowledge Distillation Beyond Model Compression. (arXiv:2310.00369v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2310.00369](http://arxiv.org/abs/2310.00369)

    本文提出了一种超越模型压缩的知识蒸馏方法，通过从轻量级教师模型中提取归纳偏差，使Vision Transformers (ViTs) 的应用成为可能。这种方法包括使用一组不同架构的教师模型来指导学生Transformer，从而有效提高学生的性能。

    

    随着计算机视觉的快速发展，Vision Transformers (ViTs) 提供了在视觉和文本领域中实现统一信息处理的诱人前景。但是由于ViTs缺乏固有的归纳偏差，它们需要大量的训练数据。为了使它们的应用实际可行，我们引入了一种创新的基于集成的蒸馏方法，从轻量级的教师模型中提取归纳偏差。以前的系统仅依靠基于卷积的教学方法。然而，这种方法将一组具有不同架构倾向的轻量级教师模型（例如卷积和非线性卷积）同时用于指导学生Transformer。由于这些独特的归纳偏差，教师模型可以从各种存储数据集中获得广泛的知识，从而提高学生的性能。我们提出的框架还涉及预先计算和存储logits，从根本上实现了非归一化的状态匹配。

    With the rapid development of computer vision, Vision Transformers (ViTs) offer the tantalizing prospect of unified information processing across visual and textual domains. But due to the lack of inherent inductive biases in ViTs, they require enormous amount of data for training. To make their applications practical, we introduce an innovative ensemble-based distillation approach distilling inductive bias from complementary lightweight teacher models. Prior systems relied solely on convolution-based teaching. However, this method incorporates an ensemble of light teachers with different architectural tendencies, such as convolution and involution, to instruct the student transformer jointly. Because of these unique inductive biases, instructors can accumulate a wide range of knowledge, even from readily identifiable stored datasets, which leads to enhanced student performance. Our proposed framework also involves precomputing and storing logits in advance, essentially the unnormalize
    
[^3]: 图像分类器的多个不同解释

    Multiple Different Explanations for Image Classifiers. (arXiv:2309.14309v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2309.14309](http://arxiv.org/abs/2309.14309)

    这篇论文介绍了一种算法和工具，可以为图像分类器的输出计算多个解释，从而提高对分类器行为的洞察力。

    

    现有的图像分类器解释工具通常只会给出一种对于图像的解释。然而，对于许多图像来说，无论是人类还是图像分类器都接受多个解释来解释图像标签。因此，限制解释的数量只有一个严重限制了对分类器行为的洞察力。在本文中，我们描述了一种算法和工具REX，用于计算黑盒图像分类器对给定图像的输出的多个解释。我们的算法基于因果理论的可靠方法。我们分析了其理论复杂性，并提供了实验结果，显示REX在ImageNet-mini基准测试中找到的多个解释比之前的工作多7倍。

    Existing explanation tools for image classifiers usually give only one single explanation for an image. For many images, however, both humans and image classifiers accept more than one explanation for the image label. Thus, restricting the number of explanations to just one severely limits the insight into the behavior of the classifier. In this paper, we describe an algorithm and a tool, REX, for computing multiple explanations of the output of a black-box image classifier for a given image. Our algorithm uses a principled approach based on causal theory. We analyse its theoretical complexity and provide experimental results showing that REX finds multiple explanations on 7 times more images than the previous work on the ImageNet-mini benchmark.
    

