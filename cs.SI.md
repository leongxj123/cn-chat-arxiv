# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Open-Set Graph Anomaly Detection via Normal Structure Regularisation](https://arxiv.org/abs/2311.06835) | 通过正常结构规范化方法，实现开放图异常检测模型对未知异常的广义检测能力 |

# 详细

[^1]: 通过正常结构规范化实现开放图异常检测

    Open-Set Graph Anomaly Detection via Normal Structure Regularisation

    [https://arxiv.org/abs/2311.06835](https://arxiv.org/abs/2311.06835)

    通过正常结构规范化方法，实现开放图异常检测模型对未知异常的广义检测能力

    

    本文考虑了一个重要的图异常检测（GAD）任务，即开放式GAD，旨在使用少量标记的训练正常节点和异常节点（称为已知异常）来检测异常节点，这些节点无法展示所有可能的推理时异常。已标记数据的可用性为GAD模型提供了关键的异常先验知识，可大大降低检测错误。然而，当前方法往往过分强调拟合已知异常，导致对未知异常（即未被标记的异常节点）的弱泛化能力。此外，它们被引入以处理欧几里德数据，未能有效捕捉GAD的重要非欧几里德特征。在这项工作中，我们提出了一种新颖的开放式GAD方法，即正常结构规范化（NSReg），以实现对未知异常的广义检测能力。

    arXiv:2311.06835v2 Announce Type: replace-cross  Abstract: This paper considers an important Graph Anomaly Detection (GAD) task, namely open-set GAD, which aims to detect anomalous nodes using a small number of labelled training normal and anomaly nodes (known as seen anomalies) that cannot illustrate all possible inference-time abnormalities. The availability of that labelled data provides crucial prior knowledge about abnormalities for GAD models, enabling substantially reduced detection errors. However, current methods tend to over-emphasise fitting the seen anomalies, leading to a weak generalisation ability to detect unseen anomalies, i.e., those that are not illustrated by the labelled anomaly nodes. Further, they were introduced to handle Euclidean data, failing to effectively capture important non-Euclidean features for GAD. In this work, we propose a novel open-set GAD approach, namely Normal Structure Regularisation (NSReg), to achieve generalised detection ability to unseen 
    

