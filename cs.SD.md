# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [What Do Language Models Hear? Probing for Auditory Representations in Language Models](https://arxiv.org/abs/2402.16998) | 通过训练一个线性探针，将语言模型中的文本表示和预训练音频模型中的声音表示联系在一起，研究发现尽管仅在原始文本上进行训练，语言模型对于一些对象的声音知识有着基于实质的编码。 |

# 详细

[^1]: 语言模型听到了什么？探究语言模型中的听觉表征

    What Do Language Models Hear? Probing for Auditory Representations in Language Models

    [https://arxiv.org/abs/2402.16998](https://arxiv.org/abs/2402.16998)

    通过训练一个线性探针，将语言模型中的文本表示和预训练音频模型中的声音表示联系在一起，研究发现尽管仅在原始文本上进行训练，语言模型对于一些对象的声音知识有着基于实质的编码。

    

    这项工作探讨了语言模型是否对物体的声音具有含义深刻且基于实质的表征。我们学习了一个线性探针，通过一个预训练的音频模型给出一个对象的声音表示，从而在给定与该对象相关的音频片段的情况下检索出该对象的正确文本表示。这个探针是通过对比损失进行训练的，推动对象的语言表示和声音表示彼此接近。在训练之后，我们测试了探针对于一些在训练中没有见过的对象的泛化能力。在不同的语言模型和音频模型中，我们发现在许多情况下探针的泛化能力超过了随机猜测的水平，这表明尽管仅在原始文本上进行训练，语言模型对于一些对象的声音知识具有基于实质的编码。

    arXiv:2402.16998v1 Announce Type: cross  Abstract: This work explores whether language models encode meaningfully grounded representations of sounds of objects. We learn a linear probe that retrieves the correct text representation of an object given a snippet of audio related to that object, where the sound representation is given by a pretrained audio model. This probe is trained via a contrastive loss that pushes the language representations and sound representations of an object to be close to one another. After training, the probe is tested on its ability to generalize to objects that were not seen during training. Across different language models and audio models, we find that the probe generalization is above chance in many cases, indicating that despite being trained only on raw text, language models encode grounded knowledge of sounds for some objects.
    

