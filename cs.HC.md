# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Loop Copilot: Conducting AI Ensembles for Music Generation and Iterative Editing.](http://arxiv.org/abs/2310.12404) | Loop Copilot是一种新型的AI音乐合奏系统，能够通过交互式多轮对话界面生成和迭代改进音乐，通过选择适当的AI模型执行任务，并在一个集中的表中保持关键属性以确保音乐的连贯性。 |
| [^2] | [EEGMatch: Learning with Incomplete Labels for Semi-Supervised EEG-based Cross-Subject Emotion Recognition.](http://arxiv.org/abs/2304.06496) | EEGMatch是一种半监督学习框架，可用于脑电情绪识别。通过基于EEG-Mixup的数据增强方法和半监督多域自适应方法，可以有效提高情绪识别准确性和稳定性。 |

# 详细

[^1]: Loop Copilot: 用于音乐生成和迭代编辑的AI合奏系统

    Loop Copilot: Conducting AI Ensembles for Music Generation and Iterative Editing. (arXiv:2310.12404v1 [cs.SD])

    [http://arxiv.org/abs/2310.12404](http://arxiv.org/abs/2310.12404)

    Loop Copilot是一种新型的AI音乐合奏系统，能够通过交互式多轮对话界面生成和迭代改进音乐，通过选择适当的AI模型执行任务，并在一个集中的表中保持关键属性以确保音乐的连贯性。

    

    创建音乐是一个迭代过程，每个阶段都需要不同的方法。然而，现有的AI音乐系统在组织多个子系统以满足不同需求方面存在不足。为了解决这个问题，我们引入了Loop Copilot，这是一个能够通过交互式、多轮对话界面生成和迭代改进音乐的新型系统。该系统使用一种大型语言模型来解释用户意图，并选择适当的AI模型进行任务执行。每个后端模型都专门针对特定任务，并将它们的输出聚合起来以满足用户的要求。为了确保音乐的连贯性，关键属性被保留在一个集中的表中。我们通过半结构化的访谈和问卷调查评估了所提出的系统的有效性，突出了它在促进音乐创作方面的实用性，以及它在更广泛应用中的潜力。

    Creating music is iterative, requiring varied methods at each stage. However, existing AI music systems fall short in orchestrating multiple subsystems for diverse needs. To address this gap, we introduce Loop Copilot, a novel system that enables users to generate and iteratively refine music through an interactive, multi-round dialogue interface. The system uses a large language model to interpret user intentions and select appropriate AI models for task execution. Each backend model is specialized for a specific task, and their outputs are aggregated to meet the user's requirements. To ensure musical coherence, essential attributes are maintained in a centralized table. We evaluate the effectiveness of the proposed system through semi-structured interviews and questionnaires, highlighting its utility not only in facilitating music creation but also its potential for broader applications.
    
[^2]: EEGMatch: 学习不完整标记的半监督脑电情绪识别

    EEGMatch: Learning with Incomplete Labels for Semi-Supervised EEG-based Cross-Subject Emotion Recognition. (arXiv:2304.06496v1 [eess.SP])

    [http://arxiv.org/abs/2304.06496](http://arxiv.org/abs/2304.06496)

    EEGMatch是一种半监督学习框架，可用于脑电情绪识别。通过基于EEG-Mixup的数据增强方法和半监督多域自适应方法，可以有效提高情绪识别准确性和稳定性。

    

    脑电图（EEG）是情绪识别的客观工具，并显示出良好的性能。然而，标签稀缺问题是该领域的主要挑战，限制了基于EEG的情绪识别的广泛应用。本文提出了一种新的半监督学习框架（EEGMatch），以利用标记和未标记的EEG数据。首先，开发了一种基于EEG-Mixup数据增强方法，以生成更多用于模型学习的有效样本。其次，提出了一种半监督的两步成对学习方法，将原型式和实例化式成对学习连接起来，其中原型式成对学习衡量EEG数据与每个情感类别的原型表示之间的全局关系，而实例化式成对学习捕捉EEG数据之间的局部内在关系。第三，引入了一种半监督的多域自适应，以对齐多个域（标记的和未标记的数据集）之间的数据表示。实验结果表明，EEGMatch在情绪识别任务中表现出比现有的半监督方法更高的准确性和稳定性。

    Electroencephalography (EEG) is an objective tool for emotion recognition and shows promising performance. However, the label scarcity problem is a main challenge in this field, which limits the wide application of EEG-based emotion recognition. In this paper, we propose a novel semi-supervised learning framework (EEGMatch) to leverage both labeled and unlabeled EEG data. First, an EEG-Mixup based data augmentation method is developed to generate more valid samples for model learning. Second, a semi-supervised two-step pairwise learning method is proposed to bridge prototype-wise and instance-wise pairwise learning, where the prototype-wise pairwise learning measures the global relationship between EEG data and the prototypical representation of each emotion class and the instance-wise pairwise learning captures the local intrinsic relationship among EEG data. Third, a semi-supervised multi-domain adaptation is introduced to align the data representation among multiple domains (labeled
    

