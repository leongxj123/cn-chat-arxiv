# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [3DCoMPaT$^{++}$: An improved Large-scale 3D Vision Dataset for Compositional Recognition](https://arxiv.org/abs/2310.18511) | 3DCoMPaT$^{++}$提出了一个大规模的多模态2D/3D数据集，包含1.6亿个渲染视图的风格化三维形状，带有详细的部件实例级别标注，用于组合识别。 |
| [^2] | [Prompting Large Language Models with Answer Heuristics for Knowledge-based Visual Question Answering.](http://arxiv.org/abs/2303.01903) | 本研究提出了一个名为Prophet的框架，使用答案启发式方式促使GPT-3解决基于知识的视觉问答问题。在特定的知识型VQA数据集上训练一个纯VQA模型，并从中提取出答案启发式，可提高模型的性能。 |

# 详细

[^1]: 3DCoMPaT$^{++}$：一个用于组合识别的改进型大规模三维视觉数据集

    3DCoMPaT$^{++}$: An improved Large-scale 3D Vision Dataset for Compositional Recognition

    [https://arxiv.org/abs/2310.18511](https://arxiv.org/abs/2310.18511)

    3DCoMPaT$^{++}$提出了一个大规模的多模态2D/3D数据集，包含1.6亿个渲染视图的风格化三维形状，带有详细的部件实例级别标注，用于组合识别。

    

    在这项工作中，我们提出了3DCoMPaT$^{++}$，这是一个包含1.6亿个以上10百万个风格化三维形状的渲染视图的多模态2D/3D数据集，这些形状在部件实例级别上进行了精心注释，并配有匹配的RGB点云、3D纹理网格、深度图和分割蒙版。3DCoMPaT$^{++}$涵盖了41个形状类别、275个细粒度部分类别和293个细粒度材料类别，这些类别可以组合应用于三维物体的各部分。我们从四个等间距视图和四个随机视图中渲染了一百万个风格化形状的子集，共计1.6亿个渲染。部件在实例级别、粗粒度和细粒度语义级别上进行了分割。我们引入了一个名为Grounded CoMPaT Recognition (GCR)的新任务，旨在共同识别和基于物体部分的材料组合。另外，我们还报告了一个数据挑战活动的结果。

    arXiv:2310.18511v2 Announce Type: replace-cross  Abstract: In this work, we present 3DCoMPaT$^{++}$, a multimodal 2D/3D dataset with 160 million rendered views of more than 10 million stylized 3D shapes carefully annotated at the part-instance level, alongside matching RGB point clouds, 3D textured meshes, depth maps, and segmentation masks. 3DCoMPaT$^{++}$ covers 41 shape categories, 275 fine-grained part categories, and 293 fine-grained material classes that can be compositionally applied to parts of 3D objects. We render a subset of one million stylized shapes from four equally spaced views as well as four randomized views, leading to a total of 160 million renderings. Parts are segmented at the instance level, with coarse-grained and fine-grained semantic levels. We introduce a new task, called Grounded CoMPaT Recognition (GCR), to collectively recognize and ground compositions of materials on parts of 3D objects. Additionally, we report the outcomes of a data challenge organized a
    
[^2]: 用答案启发式方式促使大型语言模型解决基于知识的视觉问答问题

    Prompting Large Language Models with Answer Heuristics for Knowledge-based Visual Question Answering. (arXiv:2303.01903v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2303.01903](http://arxiv.org/abs/2303.01903)

    本研究提出了一个名为Prophet的框架，使用答案启发式方式促使GPT-3解决基于知识的视觉问答问题。在特定的知识型VQA数据集上训练一个纯VQA模型，并从中提取出答案启发式，可提高模型的性能。

    

    基于知识的视觉问答需要超出图像范围的外部知识来回答问题。早期的研究从显式知识库（KBs）检索所需的知识，这经常会引入与问题无关的信息，从而限制了模型的性能。最近的研究试图将大型语言模型（即GPT-3）作为隐含式知识引擎来获取回答所需的必要知识。尽管这些方法取得了令人鼓舞的结果，但我们认为它们还没有充分发挥GPT-3的能力，因为提供的输入信息仍然不足。在本文中，我们提出了Prophet——一个概念上简单的框架，旨在通过回答启发式方式，促使GPT-3解决基于知识的VQA问题。具体来说，我们首先在特定的基于知识的VQA数据集上训练一个纯VQA模型，而不使用外部知识。之后，我们从模型中提取了两种互补的答案启发式：答案候选项。

    Knowledge-based visual question answering (VQA) requires external knowledge beyond the image to answer the question. Early studies retrieve required knowledge from explicit knowledge bases (KBs), which often introduces irrelevant information to the question, hence restricting the performance of their models. Recent works have sought to use a large language model (i.e., GPT-3) as an implicit knowledge engine to acquire the necessary knowledge for answering. Despite the encouraging results achieved by these methods, we argue that they have not fully activated the capacity of GPT-3 as the provided input information is insufficient. In this paper, we present Prophet -- a conceptually simple framework designed to prompt GPT-3 with answer heuristics for knowledge-based VQA. Specifically, we first train a vanilla VQA model on a specific knowledge-based VQA dataset without external knowledge. After that, we extract two types of complementary answer heuristics from the model: answer candidates 
    

