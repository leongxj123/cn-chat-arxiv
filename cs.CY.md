# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Disparate Impact on Group Accuracy of Linearization for Private Inference](https://arxiv.org/abs/2402.03629) | 本文研究了线性化对隐私推断中群体准确性的影响，发现减少ReLU激活函数数量会不成比例地降低少数群体的准确性，而对于多数群体则几乎没有影响。采用简单的微调步骤可以解决这个问题。 |
| [^2] | [Finding Already Debunked Narratives via Multistage Retrieval: Enabling Cross-Lingual, Cross-Dataset and Zero-Shot Learning.](http://arxiv.org/abs/2308.05680) | 本研究通过创建新的数据集、评估多语言预训练Transformer模型以及提出多阶段框架来解决了跨语言澄清检索问题。 |
| [^3] | [Causal Reasoning and Large Language Models: Opening a New Frontier for Causality.](http://arxiv.org/abs/2305.00050) | 大型语言模型在因果推理任务中取得了新的最高准确率，但是其鲁棒性仍然存在难以预测的失败模式。 |

# 详细

[^1]: 私有推断的线性化对群体准确性的不对称影响

    Disparate Impact on Group Accuracy of Linearization for Private Inference

    [https://arxiv.org/abs/2402.03629](https://arxiv.org/abs/2402.03629)

    本文研究了线性化对隐私推断中群体准确性的影响，发现减少ReLU激活函数数量会不成比例地降低少数群体的准确性，而对于多数群体则几乎没有影响。采用简单的微调步骤可以解决这个问题。

    

    确保对具有密码安全性的数据进行隐私保护的推断是一个众所周知的计算挑战。为了减轻非线性激活函数中昂贵的加密计算的瓶颈，最近的方法建议在线性神经网络中线性化目标部分的激活函数。这种技术可以显著减少运行时间，对准确性的影响往往可以忽略不计。在本文中，我们证明了这种计算优势可能导致公平性成本增加。具体而言，我们发现减少ReLU激活函数数量会不成比例地降低少数群体的准确性，而对于多数群体则几乎没有影响。为了解释这些观察结果，我们在对决策边界性质进行限制性假设的基础上提供了数学解释，同时还展示了这个问题在广泛使用的数据集和体系结构中的普遍性。最后，我们展示了如何通过简单的程序改变线性模型的微调步骤来解决这个问题。

    Ensuring privacy-preserving inference on cryptographically secure data is a well-known computational challenge. To alleviate the bottleneck of costly cryptographic computations in non-linear activations, recent methods have suggested linearizing a targeted portion of these activations in neural networks. This technique results in significantly reduced runtimes with often negligible impacts on accuracy. In this paper, we demonstrate that such computational benefits may lead to increased fairness costs. Specifically, we find that reducing the number of ReLU activations disproportionately decreases the accuracy for minority groups compared to majority groups. To explain these observations, we provide a mathematical interpretation under restricted assumptions about the nature of the decision boundary, while also showing the prevalence of this problem across widely used datasets and architectures. Finally, we show how a simple procedure altering the fine-tuning step for linearized models ca
    
[^2]: 通过多阶段检索找到已经被澄清的叙述：实现跨语言、跨数据集和零样本学习

    Finding Already Debunked Narratives via Multistage Retrieval: Enabling Cross-Lingual, Cross-Dataset and Zero-Shot Learning. (arXiv:2308.05680v1 [cs.CL])

    [http://arxiv.org/abs/2308.05680](http://arxiv.org/abs/2308.05680)

    本研究通过创建新的数据集、评估多语言预训练Transformer模型以及提出多阶段框架来解决了跨语言澄清检索问题。

    

    检索已经被澄清的叙述的任务旨在检测已经经过事实核查的故事。成功检测到已被澄清的声明不仅减少了专业事实核查人员的手动努力，还可以有助于减缓虚假信息的传播。由于缺乏可用数据，这是一个研究不足的问题，特别是在考虑跨语言任务时，即在检查的在线帖子的语言与事实核查文章的语言不同的情况下进行检索。本文通过以下方式填补了这一空白：（i）创建了一个新颖的数据集，以允许对已被澄清的叙述进行跨语言检索的研究，使用推文作为对事实核查文章数据库的查询；（ii）展示了一个全面的实验，以评估经过微调和现成的多语言预训练Transformer模型在这个任务上的性能；（iii）提出了一个新颖的多阶段框架，将这个跨语言澄清检索问题划分为不同的阶段。

    The task of retrieving already debunked narratives aims to detect stories that have already been fact-checked. The successful detection of claims that have already been debunked not only reduces the manual efforts of professional fact-checkers but can also contribute to slowing the spread of misinformation. Mainly due to the lack of readily available data, this is an understudied problem, particularly when considering the cross-lingual task, i.e. the retrieval of fact-checking articles in a language different from the language of the online post being checked. This paper fills this gap by (i) creating a novel dataset to enable research on cross-lingual retrieval of already debunked narratives, using tweets as queries to a database of fact-checking articles; (ii) presenting an extensive experiment to benchmark fine-tuned and off-the-shelf multilingual pre-trained Transformer models for this task; and (iii) proposing a novel multistage framework that divides this cross-lingual debunk ret
    
[^3]: 因果推理与大型语言模型：开启因果研究的新篇章

    Causal Reasoning and Large Language Models: Opening a New Frontier for Causality. (arXiv:2305.00050v1 [cs.AI])

    [http://arxiv.org/abs/2305.00050](http://arxiv.org/abs/2305.00050)

    大型语言模型在因果推理任务中取得了新的最高准确率，但是其鲁棒性仍然存在难以预测的失败模式。

    

    大型语言模型的因果能力备受争议，并且对将其应用于医学、科学、法律和政策等具有社会影响力的领域具有重要意义。我们进一步探讨了LLMs及其因果推理的区别，以及潜在的建构和测量效度威胁。基于GPT-3.5和4的算法在多个因果基准测试上取得了新的最高准确率。与此同时，LLMs展示了难以预测的失败模式，我们提供了一些技术来解释它们的鲁棒性。

    The causal capabilities of large language models (LLMs) is a matter of significant debate, with critical implications for the use of LLMs in societally impactful domains such as medicine, science, law, and policy. We further our understanding of LLMs and their causal implications, considering the distinctions between different types of causal reasoning tasks, as well as the entangled threats of construct and measurement validity. LLM-based methods establish new state-of-the-art accuracies on multiple causal benchmarks. Algorithms based on GPT-3.5 and 4 outperform existing algorithms on a pairwise causal discovery task (97%, 13 points gain), counterfactual reasoning task (92%, 20 points gain), and actual causality (86% accuracy in determining necessary and sufficient causes in vignettes). At the same time, LLMs exhibit unpredictable failure modes and we provide some techniques to interpret their robustness.  Crucially, LLMs perform these causal tasks while relying on sources of knowledg
    

