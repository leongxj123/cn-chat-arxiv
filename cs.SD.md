# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Loop Copilot: Conducting AI Ensembles for Music Generation and Iterative Editing.](http://arxiv.org/abs/2310.12404) | Loop Copilot是一种新型的AI音乐合奏系统，能够通过交互式多轮对话界面生成和迭代改进音乐，通过选择适当的AI模型执行任务，并在一个集中的表中保持关键属性以确保音乐的连贯性。 |
| [^2] | [MelHuBERT: A simplified HuBERT on Mel spectrograms.](http://arxiv.org/abs/2211.09944) | MelHuBERT是基于Mel频谱图的简化版HuBERT模型，通过改进损失函数、输入表示和多阶段训练，在语音识别方面取得了有利表现，节省了31.2%的预训练时间和33.5%的计算资源。 |

# 详细

[^1]: Loop Copilot: 用于音乐生成和迭代编辑的AI合奏系统

    Loop Copilot: Conducting AI Ensembles for Music Generation and Iterative Editing. (arXiv:2310.12404v1 [cs.SD])

    [http://arxiv.org/abs/2310.12404](http://arxiv.org/abs/2310.12404)

    Loop Copilot是一种新型的AI音乐合奏系统，能够通过交互式多轮对话界面生成和迭代改进音乐，通过选择适当的AI模型执行任务，并在一个集中的表中保持关键属性以确保音乐的连贯性。

    

    创建音乐是一个迭代过程，每个阶段都需要不同的方法。然而，现有的AI音乐系统在组织多个子系统以满足不同需求方面存在不足。为了解决这个问题，我们引入了Loop Copilot，这是一个能够通过交互式、多轮对话界面生成和迭代改进音乐的新型系统。该系统使用一种大型语言模型来解释用户意图，并选择适当的AI模型进行任务执行。每个后端模型都专门针对特定任务，并将它们的输出聚合起来以满足用户的要求。为了确保音乐的连贯性，关键属性被保留在一个集中的表中。我们通过半结构化的访谈和问卷调查评估了所提出的系统的有效性，突出了它在促进音乐创作方面的实用性，以及它在更广泛应用中的潜力。

    Creating music is iterative, requiring varied methods at each stage. However, existing AI music systems fall short in orchestrating multiple subsystems for diverse needs. To address this gap, we introduce Loop Copilot, a novel system that enables users to generate and iteratively refine music through an interactive, multi-round dialogue interface. The system uses a large language model to interpret user intentions and select appropriate AI models for task execution. Each backend model is specialized for a specific task, and their outputs are aggregated to meet the user's requirements. To ensure musical coherence, essential attributes are maintained in a centralized table. We evaluate the effectiveness of the proposed system through semi-structured interviews and questionnaires, highlighting its utility not only in facilitating music creation but also its potential for broader applications.
    
[^2]: MelHuBERT: 一种基于Mel频谱图的简化HuBERT模型

    MelHuBERT: A simplified HuBERT on Mel spectrograms. (arXiv:2211.09944v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.09944](http://arxiv.org/abs/2211.09944)

    MelHuBERT是基于Mel频谱图的简化版HuBERT模型，通过改进损失函数、输入表示和多阶段训练，在语音识别方面取得了有利表现，节省了31.2%的预训练时间和33.5%的计算资源。

    

    自监督模型在学习语音表示方面取得了巨大的成功，可以推广到各种下游任务。然而，大多数自监督模型需要大量的计算资源和多个GPU来进行训练，从而严重限制了自监督学习的发展。为了减少训练的计算量，我们重新审视了HuBERT的训练方法，这是一个非常成功的自监督模型。我们改进并简化了几个关键组成部分，包括损失函数、输入表示和多阶段训练。我们的模型MelHuBERT在音素识别、说话人识别和自动语音识别方面均能取得较好的性能，同时节省了31.2%的预训练时间，或等效地每秒语音节省了33.5%的MACs。代码和预训练模型可在https://github.com/nervjack2/MelHuBERT中获得。

    Self-supervised models have had great success in learning speech representations that can generalize to various downstream tasks. However, most self-supervised models require a large amount of compute and multiple GPUs to train, significantly hampering the development of self-supervised learning. In an attempt to reduce the computation of training, we revisit the training of HuBERT, a highly successful self-supervised model. We improve and simplify several key components, including the loss function, input representation, and training in multiple stages. Our model, MelHuBERT, is able to achieve favorable performance on phone recognition, speaker identification, and automatic speech recognition against HuBERT, while saving 31.2% of the pre-training time, or equivalently 33.5% MACs per one second speech. The code and pre-trained models are available in https://github.com/nervjack2/MelHuBERT.
    

