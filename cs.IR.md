# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Aligning and Training Framework for Multimodal Recommendations](https://arxiv.org/abs/2403.12384) | 提出了一种名为AlignRec的对齐和训练框架，用于解决多模态推荐中的不对齐问题，通过将推荐目标分解为三个对齐部分，实现内容内部对齐、内容与分类ID之间的对齐以及用户和项目之间的对齐。 |

# 详细

[^1]: 一种用于多模态推荐的对齐和训练框架

    An Aligning and Training Framework for Multimodal Recommendations

    [https://arxiv.org/abs/2403.12384](https://arxiv.org/abs/2403.12384)

    提出了一种名为AlignRec的对齐和训练框架，用于解决多模态推荐中的不对齐问题，通过将推荐目标分解为三个对齐部分，实现内容内部对齐、内容与分类ID之间的对齐以及用户和项目之间的对齐。

    

    随着多媒体应用的发展，多模态推荐正在发挥着重要作用，因为它们可以利用超越用户交互的丰富上下文。现有方法主要将多模态信息视为辅助，用于帮助学习ID特征；然而，多模态内容特征和ID特征之间存在语义差距，直接将多模态信息作为辅助使用会导致用户和项目表示的不对齐。本文首先系统地研究了多模态推荐中的不对齐问题，并提出了一种名为AlignRec的解决方案。在AlignRec中，推荐目标被分解为三个对齐部分，即内容内部对齐，内容与分类ID之间的对齐以及用户和项目之间的对齐。每个对齐部分都由特定的目标函数来表征，并整合到我们的多模态推荐中。

    arXiv:2403.12384v1 Announce Type: cross  Abstract: With the development of multimedia applications, multimodal recommendations are playing an essential role, as they can leverage rich contexts beyond user interactions. Existing methods mainly regard multimodal information as an auxiliary, using them to help learn ID features; however, there exist semantic gaps among multimodal content features and ID features, for which directly using multimodal information as an auxiliary would lead to misalignment in representations of users and items. In this paper, we first systematically investigate the misalignment issue in multimodal recommendations, and propose a solution named AlignRec. In AlignRec, the recommendation objective is decomposed into three alignments, namely alignment within contents, alignment between content and categorical ID, and alignment between users and items. Each alignment is characterized by a specific objective function and is integrated into our multimodal recommendat
    

