# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DIRECT: Deep Active Learning under Imbalance and Label Noise](https://rss.arxiv.org/abs/2312.09196) | 这篇论文提出了一种名为DIRECT的算法，用于处理不平衡和标签噪声下的深度主动学习问题。通过确定类别分割阈值并标记最不确定且离其最近的示例，该算法能够有效解决罕见类和少数类的性能问题，并具有批次标记和对标签噪声的容忍能力。 |
| [^2] | [Mitigating Object Hallucination in Large Vision-Language Models via Classifier-Free Guidance](https://arxiv.org/abs/2402.08680) | 本文介绍了一种名为MARINE的框架，用于通过无分类器引导来减少大型视觉语言模型的物体幻觉。该框架无需训练或API访问，并通过集成视觉模型和引入额外的物体基础特征来提高模型的生成精确性和效率。 |

# 详细

[^1]: DIRECT: 处理不平衡和标签噪声下的深度主动学习

    DIRECT: Deep Active Learning under Imbalance and Label Noise

    [https://rss.arxiv.org/abs/2312.09196](https://rss.arxiv.org/abs/2312.09196)

    这篇论文提出了一种名为DIRECT的算法，用于处理不平衡和标签噪声下的深度主动学习问题。通过确定类别分割阈值并标记最不确定且离其最近的示例，该算法能够有效解决罕见类和少数类的性能问题，并具有批次标记和对标签噪声的容忍能力。

    

    在现实世界的机器学习应用中，类别不平衡是一个普遍存在的问题，通常会导致罕见和少数类的性能较差。在大量未标记数据的情况下，主动学习可能是解决该问题的最有效技术，它从根本上采集更平衡和具有信息量的标记示例进行注释。标签噪声是数据注释任务中另一个常见问题，对于主动学习方法来说尤其具有挑战性。本文首次研究了在类别不平衡和标签噪声下的主动学习。我们提出了一种新颖的算法，能够稳健地确定类别分割阈值并标记最不确定且离其最近的示例。通过将问题简化为一维主动学习，我们的算法DIRECT能够利用经典的主动学习文献来解决批次标记和对标签噪声的容忍等问题。我们在大量实验中展示了算法的性能。

    Class imbalance is a prevalent issue in real world machine learning applications, often leading to poor performance in rare and minority classes. With an abundance of wild unlabeled data, active learning is perhaps the most effective technique in solving the problem at its root -- collecting a more balanced and informative set of labeled examples during annotation. Label noise is another common issue in data annotation jobs, which is especially challenging for active learning methods. In this work, we conduct the first study of active learning under both class imbalance and label noise. We propose a novel algorithm that robustly identifies the class separation threshold and annotates the most uncertain examples that are closest from it. Through a novel reduction to one-dimensional active learning, our algorithm DIRECT is able to leverage the classic active learning literature to address issues such as batch labeling and tolerance towards label noise. We present extensive experiments on
    
[^2]: 通过无分类器引导来减轻大型视觉语言模型中的物体幻觉

    Mitigating Object Hallucination in Large Vision-Language Models via Classifier-Free Guidance

    [https://arxiv.org/abs/2402.08680](https://arxiv.org/abs/2402.08680)

    本文介绍了一种名为MARINE的框架，用于通过无分类器引导来减少大型视觉语言模型的物体幻觉。该框架无需训练或API访问，并通过集成视觉模型和引入额外的物体基础特征来提高模型的生成精确性和效率。

    

    大型视觉语言模型（LVLM）的进展越来越突出了它们在图像中产生虚假物体的严重问题。为了解决这个问题，先前的研究着重于使用特殊策划的数据集或强大的LLM（例如GPT-3.5）来纠正LVLM的输出。然而，这些方法要求昂贵的训练/微调或API访问先进的LLM来在生成后纠正模型的输出。在本文中，我们通过引入一个名为通过无分类器引导缓解幻觉的框架（MARINE）来解决这个挑战，该框架既无需训练也无需API访问，可以在生成过程中有效地减少物体幻觉。具体而言，MARINE通过集成现有的开源视觉模型丰富LVLM的视觉语境，并使用无分类器引导来整合额外的物体基础特征，以提高LVLM生成的精确性。

    The advancement of Large Vision-Language Models (LVLMs) has increasingly highlighted the critical issue of their tendency to hallucinate non-existing objects in the images. To address this issue, previous works focused on using specially curated datasets or powerful LLMs (e.g., GPT-3.5) to rectify the outputs of LVLMs. However, these approaches require either expensive training/fine-tuning or API access to advanced LLMs to correct the model's output post-generation. In this paper, we tackle this challenge by introducing a framework called Mitigating hallucinAtion via classifieR-Free guIdaNcE (MARINE), which is both training-free and API-free, and can effectively and efficiently reduce object hallucinations during the generation process. Specifically, MARINE enriches the visual context of LVLMs by integrating existing open-source vision models, and employs classifier-free guidance to incorporate the additional object grounding features to improve the precision of LVLMs' generations. Thr
    

