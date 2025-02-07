# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Collage Prompting: Budget-Friendly Visual Recognition with GPT-4V](https://arxiv.org/abs/2403.11468) | 通过引入Collage Prompting方法，我们实现了与GPT-4V合作的经济可行的视觉识别方法，通过优化图像排列顺序获得最大的识别准确性。 |
| [^2] | [A Survey on 3D Skeleton Based Person Re-Identification: Approaches, Designs, Challenges, and Future Directions.](http://arxiv.org/abs/2401.15296) | 本文通过对当前基于3D骨架的人员再识别方法、模型设计、挑战和未来方向的系统调研，填补了相关研究总结的空白。 |

# 详细

[^1]: Collage Prompting: 与GPT-4V合作的经济可行的视觉识别

    Collage Prompting: Budget-Friendly Visual Recognition with GPT-4V

    [https://arxiv.org/abs/2403.11468](https://arxiv.org/abs/2403.11468)

    通过引入Collage Prompting方法，我们实现了与GPT-4V合作的经济可行的视觉识别方法，通过优化图像排列顺序获得最大的识别准确性。

    

    最近生成式人工智能的进展表明，通过采用视觉提示，GPT-4V可以在图像识别任务中展现出显著的熟练度。尽管其令人印象深刻的能力，但与GPT-4V的推断相关的财务成本构成了其广泛应用的重大障碍。为了解决这一挑战，我们的研究引入了Collage Prompting，这是一种经济实惠的提示方法，将多个图像连接成单个视觉输入。借助拼贴提示，GPT-4V可以同时在多幅图像上执行图像识别。基于GPT-4V的图像识别准确性与拼贴提示中图像顺序明显变化的观察，我们的方法进一步学习优化图像安排以获得最大的识别准确性。训练了一个图预测器来指示每个拼贴提示的准确性，然后我们提出了一种优化方法来导航搜索空间。

    arXiv:2403.11468v1 Announce Type: cross  Abstract: Recent advancements in generative AI have suggested that by taking visual prompt, GPT-4V can demonstrate significant proficiency in image recognition task. Despite its impressive capabilities, the financial cost associated with GPT-4V's inference presents a substantial barrier for its wide use. To address this challenge, our work introduces Collage Prompting, a budget-friendly prompting approach that concatenates multiple images into a single visual input. With collage prompt, GPT-4V is able to perform image recognition on several images simultaneously. Based on the observation that the accuracy of GPT-4V's image recognition varies significantly with the order of images within the collage prompt, our method further learns to optimize the arrangement of images for maximum recognition accuracy. A graph predictor is trained to indicate the accuracy of each collage prompt, then we propose an optimization method to navigate the search space
    
[^2]: 基于3D骨架的人员再识别：方法、设计、挑战和未来方向的综述

    A Survey on 3D Skeleton Based Person Re-Identification: Approaches, Designs, Challenges, and Future Directions. (arXiv:2401.15296v1 [cs.CV])

    [http://arxiv.org/abs/2401.15296](http://arxiv.org/abs/2401.15296)

    本文通过对当前基于3D骨架的人员再识别方法、模型设计、挑战和未来方向的系统调研，填补了相关研究总结的空白。

    

    通过3D骨架进行人员再识别是一个重要的新兴研究领域，引起了模式识别社区的极大兴趣。近年来，针对骨架建模和特征学习中突出问题，已经提出了许多具有独特优势的基于3D骨架的人员再识别（SRID）方法。尽管近年来取得了一些进展，但据我们所知，目前还没有对这些研究及其挑战进行综合总结。因此，本文通过对当前SRID方法、模型设计、挑战和未来方向的系统调研，试图填补这一空白。具体而言，我们首先定义了SRID问题，并提出了一个SRID研究的分类体系，总结了常用的基准数据集、常用的模型架构，并对不同方法的特点进行了分析评价。然后，我们详细阐述了SRID模型的设计原则。

    Person re-identification via 3D skeletons is an important emerging research area that triggers great interest in the pattern recognition community. With distinctive advantages for many application scenarios, a great diversity of 3D skeleton based person re-identification (SRID) methods have been proposed in recent years, effectively addressing prominent problems in skeleton modeling and feature learning. Despite recent advances, to the best of our knowledge, little effort has been made to comprehensively summarize these studies and their challenges. In this paper, we attempt to fill this gap by providing a systematic survey on current SRID approaches, model designs, challenges, and future directions. Specifically, we first formulate the SRID problem, and propose a taxonomy of SRID research with a summary of benchmark datasets, commonly-used model architectures, and an analytical review of different methods' characteristics. Then, we elaborate on the design principles of SRID models fro
    

