# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Hierarchical Classification of Research Fields in the "Web of Science" Using Deep Learning.](http://arxiv.org/abs/2302.00390) | 本文提出了一个使用深度学习进行层次分类的系统，可以自动将学术出版物通过抽象进行分类，实现了对研究活动在不同层次结构中的全面分类，并允许跨学科和跨领域的单标签和多标签分类。 |

# 详细

[^1]: 使用深度学习在"Web of Science"中对研究领域进行层次分类

    Hierarchical Classification of Research Fields in the "Web of Science" Using Deep Learning. (arXiv:2302.00390v2 [cs.DL] UPDATED)

    [http://arxiv.org/abs/2302.00390](http://arxiv.org/abs/2302.00390)

    本文提出了一个使用深度学习进行层次分类的系统，可以自动将学术出版物通过抽象进行分类，实现了对研究活动在不同层次结构中的全面分类，并允许跨学科和跨领域的单标签和多标签分类。

    

    本文提出了一个层次分类系统，通过使用抽象将学术出版物自动分类到三级层次标签集（学科、领域、子领域）中，以多类别设置进行分类。该系统可以通过文章的知识生产和引用的影响，对研究活动在所述层次结构中进行全面的分类，并允许这些活动被归为多个类别。该分类系统在Microsoft Academic Graph (版本2018-05-17)的160 million份摘要片段中区分了44个学科、718个领域和1,485个子领域。我们以模块化和分布式方式进行批量训练，以解决和允许跨学科和跨领域的单标签和多标签分类。总共，我们在所有考虑的模型（卷积神经网络，循环神经网络，变形器）中进行了3,140次实验。分类准确率> 90％。

    This paper presents a hierarchical classification system that automatically categorizes a scholarly publication using its abstract into a three-tier hierarchical label set (discipline, field, subfield) in a multi-class setting. This system enables a holistic categorization of research activities in the mentioned hierarchy in terms of knowledge production through articles and impact through citations, permitting those activities to fall into multiple categories. The classification system distinguishes 44 disciplines, 718 fields and 1,485 subfields among 160 million abstract snippets in Microsoft Academic Graph (version 2018-05-17). We used batch training in a modularized and distributed fashion to address and allow for interdisciplinary and interfield classifications in single-label and multi-label settings. In total, we have conducted 3,140 experiments in all considered models (Convolutional Neural Networks, Recurrent Neural Networks, Transformers). The classification accuracy is > 90%
    

