# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diverse Neural Audio Embeddings -- Bringing Features back !.](http://arxiv.org/abs/2309.08751) | 本文通过在音频分类任务中学习多样化的特征表示，包括领域特定的音高、音色和神经表示，以及端到端架构，为学习稳健、多样化的表示铺平了道路，并显著提高了性能。 |

# 详细

[^1]: 多样的神经音频嵌入 - 恢复特征！

    Diverse Neural Audio Embeddings -- Bringing Features back !. (arXiv:2309.08751v1 [cs.SD])

    [http://arxiv.org/abs/2309.08751](http://arxiv.org/abs/2309.08751)

    本文通过在音频分类任务中学习多样化的特征表示，包括领域特定的音高、音色和神经表示，以及端到端架构，为学习稳健、多样化的表示铺平了道路，并显著提高了性能。

    

    随着现代人工智能架构的出现，从端到端的架构开始流行。这种转变导致了神经架构在没有领域特定偏见/知识的情况下进行训练，根据任务进行优化。本文中，我们通过多样的特征表示（在本例中是领域特定的）学习音频嵌入。对于涉及数百种声音分类的情况，我们学习分别针对音高、音色和神经表示等多样的音频属性建立稳健的嵌入，同时也通过端到端架构进行学习。我们观察到手工制作的嵌入，例如基于音高和音色的嵌入，虽然单独使用时无法击败完全端到端的表示，但将这些嵌入与端到端嵌入相结合可以显著提高性能。这项工作将为在端到端模型中引入一些领域专业知识来学习稳健、多样化的表示铺平道路，并超越仅训练端到端模型的性能。

    With the advent of modern AI architectures, a shift has happened towards end-to-end architectures. This pivot has led to neural architectures being trained without domain-specific biases/knowledge, optimized according to the task. We in this paper, learn audio embeddings via diverse feature representations, in this case, domain-specific. For the case of audio classification over hundreds of categories of sound, we learn robust separate embeddings for diverse audio properties such as pitch, timbre, and neural representation, along with also learning it via an end-to-end architecture. We observe handcrafted embeddings, e.g., pitch and timbre-based, although on their own, are not able to beat a fully end-to-end representation, yet adding these together with end-to-end embedding helps us, significantly improve performance. This work would pave the way to bring some domain expertise with end-to-end models to learn robust, diverse representations, surpassing the performance of just training 
    

