# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Corrective Machine Unlearning](https://arxiv.org/abs/2402.14015) | 该论文通过形式化“修正机器消除”来解决受未知操纵影响的数据对训练模型的影响问题，可能仅知道一部分受影响样本。发现纠正消除问题与传统以隐私为导向的消除方法有显著不同的要求。 |
| [^2] | [TMI! Finetuned Models Leak Private Information from their Pretraining Data.](http://arxiv.org/abs/2306.01181) | 本文提出了一种新的会员推断威胁模型TMI，用于评估微调模型对预训练数据的泄露，突显了在使用预训练模型进行迁移学习中存在的隐私风险，并需要对机器学习中的隐私进行更严格的评估。 |

# 详细

[^1]: 修正机器消除

    Corrective Machine Unlearning

    [https://arxiv.org/abs/2402.14015](https://arxiv.org/abs/2402.14015)

    该论文通过形式化“修正机器消除”来解决受未知操纵影响的数据对训练模型的影响问题，可能仅知道一部分受影响样本。发现纠正消除问题与传统以隐私为导向的消除方法有显著不同的要求。

    

    机器学习模型越来越面临数据完整性挑战，因为它们使用了大规模的从互联网中获取的训练数据集。本文研究了如果模型开发者发现某些数据被篡改或错误，他们可以采取什么措施。这些被篡改的数据会导致不利影响，如容易受到后门样本的攻击、系统性偏见，以及在某些输入领域的准确度降低。通常，并非所有被篡改的训练样本都是已知的，而只有一小部分代表性的受影响数据被标记。

    arXiv:2402.14015v1 Announce Type: cross  Abstract: Machine Learning models increasingly face data integrity challenges due to the use of large-scale training datasets drawn from the internet. We study what model developers can do if they detect that some data was manipulated or incorrect. Such manipulated data can cause adverse effects like vulnerability to backdoored samples, systematic biases, and in general, reduced accuracy on certain input domains. Often, all manipulated training samples are not known, and only a small, representative subset of the affected data is flagged.   We formalize "Corrective Machine Unlearning" as the problem of mitigating the impact of data affected by unknown manipulations on a trained model, possibly knowing only a subset of impacted samples. We demonstrate that the problem of corrective unlearning has significantly different requirements from traditional privacy-oriented unlearning. We find most existing unlearning methods, including the gold-standard
    
[^2]: 过拟合的模型会泄露预训练数据的隐私信息

    TMI! Finetuned Models Leak Private Information from their Pretraining Data. (arXiv:2306.01181v1 [cs.LG])

    [http://arxiv.org/abs/2306.01181](http://arxiv.org/abs/2306.01181)

    本文提出了一种新的会员推断威胁模型TMI，用于评估微调模型对预训练数据的泄露，突显了在使用预训练模型进行迁移学习中存在的隐私风险，并需要对机器学习中的隐私进行更严格的评估。

    

    迁移学习已成为机器学习中越来越流行的技术，用于利用为一个任务训练的预训练模型来协助构建相关任务的微调模型。该范例在隐私机器学习方面尤其受欢迎，其中预训练模型被认为是公开的，只有微调数据被视为敏感的。然而，有理由认为用于预训练的数据仍然是敏感的，因此必须了解微调模型泄露有关预训练数据的信息量。本文提出了一种新的会员推理威胁模型，其中对手只能访问已经微调好的模型，并想推断预训练数据的成员资格。为了实现这个威胁模型，我们实施了一种新型的基于元分类器的攻击TMI，它利用了在下游任务中记忆的预训练样本对预测的影响。我们在视觉和自然语言处理任务上评估了TMI，并表明它在仅使用微调模型的情况下实现了高精度的推断预训练数据的成员资格。我们的结果突显了在迁移学习中使用预训练模型可能存在的隐私风险，以及需要对机器学习中的隐私进行更严格的评估的需求。

    Transfer learning has become an increasingly popular technique in machine learning as a way to leverage a pretrained model trained for one task to assist with building a finetuned model for a related task. This paradigm has been especially popular for privacy in machine learning, where the pretrained model is considered public, and only the data for finetuning is considered sensitive. However, there are reasons to believe that the data used for pretraining is still sensitive, making it essential to understand how much information the finetuned model leaks about the pretraining data. In this work we propose a new membership-inference threat model where the adversary only has access to the finetuned model and would like to infer the membership of the pretraining data. To realize this threat model, we implement a novel metaclassifier-based attack, TMI, that leverages the influence of memorized pretraining samples on predictions in the downstream task. We evaluate TMI on both vision and na
    

