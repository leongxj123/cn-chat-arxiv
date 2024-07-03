# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards Unsupervised Question Answering System with Multi-level Summarization for Legal Text](https://arxiv.org/abs/2403.13107) | 提出了一种无监督问答系统，通过多级总结法对法律文本进行处理，实现了F1分数的显著提升 |
| [^2] | [Farsight: Fostering Responsible AI Awareness During AI Application Prototyping](https://arxiv.org/abs/2402.15350) | Farsight是一个新颖的实地交互工具，帮助人们在设计AI应用原型时识别潜在危害，用户研究表明使用Farsight后，AI原型设计者能够更好地独立识别与提示相关的潜在危害。 |
| [^3] | [Shall We Talk: Exploring Spontaneous Collaborations of Competing LLM Agents](https://arxiv.org/abs/2402.12327) | 该研究揭示了LLM代理甚至在竞争环境中也能自发形成合作关系的能力，验证了计算社会科学的愿景，表明LLM代理可以用于模拟人类社会互动，包括自发合作的互动，为社会现象提供洞察。 |
| [^4] | [A Survey on Safe Multi-Modal Learning System](https://arxiv.org/abs/2402.05355) | 这项研究提出了第一个多模态学习系统安全的分类法，对当前发展状态下的关键限制进行了审查，并提出了未来研究的潜在方向。 |

# 详细

[^1]: 面向法律文本的多级总结无监督问答系统

    Towards Unsupervised Question Answering System with Multi-level Summarization for Legal Text

    [https://arxiv.org/abs/2403.13107](https://arxiv.org/abs/2403.13107)

    提出了一种无监督问答系统，通过多级总结法对法律文本进行处理，实现了F1分数的显著提升

    

    这篇论文总结了团队SCaLAR在SemEval-2024任务5上的工作：民事程序中的法律论证推理。为了解决这个二元分类任务，由于涉及到的法律文本的复杂性而令人望而却步，我们提出了一种简单而又新颖的基于相似度和距离的无监督方法来生成标签。此外，我们探索了使用集成特征（包括CNN、GRU和LSTM）的多级Legal-Bert嵌入的融合。为了解决数据集中法律解释的冗长性，我们引入了基于T5的分段摘要，成功地保留了关键信息，提升了模型的性能。我们的无监督系统在开发集上见证了macro F1分数增加了20个百分点，在测试集上增加了10个百分点，考虑到其简单的架构，这是令人鼓舞的。

    arXiv:2403.13107v1 Announce Type: new  Abstract: This paper summarizes Team SCaLAR's work on SemEval-2024 Task 5: Legal Argument Reasoning in Civil Procedure. To address this Binary Classification task, which was daunting due to the complexity of the Legal Texts involved, we propose a simple yet novel similarity and distance-based unsupervised approach to generate labels. Further, we explore the Multi-level fusion of Legal-Bert embeddings using ensemble features, including CNN, GRU, and LSTM. To address the lengthy nature of Legal explanation in the dataset, we introduce T5-based segment-wise summarization, which successfully retained crucial information, enhancing the model's performance. Our unsupervised system witnessed a 20-point increase in macro F1-score on the development set and a 10-point increase on the test set, which is promising given its uncomplicated architecture.
    
[^2]: Farsight：在AI应用原型设计过程中培养负责任的AI意识

    Farsight: Fostering Responsible AI Awareness During AI Application Prototyping

    [https://arxiv.org/abs/2402.15350](https://arxiv.org/abs/2402.15350)

    Farsight是一个新颖的实地交互工具，帮助人们在设计AI应用原型时识别潜在危害，用户研究表明使用Farsight后，AI原型设计者能够更好地独立识别与提示相关的潜在危害。

    

    大型语言模型（LLM）的提示驱动界面使得原型设计和构建AI应用比以往任何时候都更容易。然而，识别可能在AI应用中出现的潜在危害仍然是一个挑战，特别是在基于提示的原型设计过程中。为了解决这一问题，我们提出了一种新颖的实地交互工具Farsight，帮助人们识别他们正在设计原型的AI应用中可能出现的潜在危害。根据用户的提示，Farsight突出显示了与相关AI事件有关的新闻文章，并允许用户探索和编辑LLM生成的用例、利益相关者和危害。我们报告了与10位AI原型设计者进行的共同设计研究的设计见解，以及与42位AI原型设计者进行的用户研究结果。在使用Farsight后，我们用户研究中的AI原型设计者能够更好地独立识别与提示相关的潜在危害，并发现我们的工具比现有资源更有用且更易于使用。

    arXiv:2402.15350v1 Announce Type: cross  Abstract: Prompt-based interfaces for Large Language Models (LLMs) have made prototyping and building AI-powered applications easier than ever before. However, identifying potential harms that may arise from AI applications remains a challenge, particularly during prompt-based prototyping. To address this, we present Farsight, a novel in situ interactive tool that helps people identify potential harms from the AI applications they are prototyping. Based on a user's prompt, Farsight highlights news articles about relevant AI incidents and allows users to explore and edit LLM-generated use cases, stakeholders, and harms. We report design insights from a co-design study with 10 AI prototypers and findings from a user study with 42 AI prototypers. After using Farsight, AI prototypers in our user study are better able to independently identify potential harms associated with a prompt and find our tool more useful and usable than existing resources. T
    
[^3]: 我们应该交流吗：探索竞争LLM代理之间的自发合作

    Shall We Talk: Exploring Spontaneous Collaborations of Competing LLM Agents

    [https://arxiv.org/abs/2402.12327](https://arxiv.org/abs/2402.12327)

    该研究揭示了LLM代理甚至在竞争环境中也能自发形成合作关系的能力，验证了计算社会科学的愿景，表明LLM代理可以用于模拟人类社会互动，包括自发合作的互动，为社会现象提供洞察。

    

    最近的进展表明，由大型语言模型（LLMs）驱动的代理具有模拟人类行为和社会动态的能力。然而，尚未研究LLM代理在没有明确指令的情况下自发建立合作关系的潜力。为了弥补这一空白，我们进行了三项案例研究，揭示了LLM代理甚至在竞争环境中也能自发形成合作关系的能力。这一发现不仅展示了LLM代理模拟人类社会中竞争与合作的能力，也验证了计算社会科学的一个有前途的愿景。具体来说，这表明LLM代理可以用于建模人类社会互动，包括那些自发合作的互动，从而提供对社会现象的洞察。这项研究的源代码可在https://github.com/wuzengqing001225/SABM_ShallWe 找到。

    arXiv:2402.12327v1 Announce Type: new  Abstract: Recent advancements have shown that agents powered by large language models (LLMs) possess capabilities to simulate human behaviors and societal dynamics. However, the potential for LLM agents to spontaneously establish collaborative relationships in the absence of explicit instructions has not been studied. To address this gap, we conduct three case studies, revealing that LLM agents are capable of spontaneously forming collaborations even within competitive settings. This finding not only demonstrates the capacity of LLM agents to mimic competition and cooperation in human societies but also validates a promising vision of computational social science. Specifically, it suggests that LLM agents could be utilized to model human social interactions, including those with spontaneous collaborations, thus offering insights into social phenomena. The source codes for this study are available at https://github.com/wuzengqing001225/SABM_ShallWe
    
[^4]: 安全多模态学习系统调研

    A Survey on Safe Multi-Modal Learning System

    [https://arxiv.org/abs/2402.05355](https://arxiv.org/abs/2402.05355)

    这项研究提出了第一个多模态学习系统安全的分类法，对当前发展状态下的关键限制进行了审查，并提出了未来研究的潜在方向。

    

    随着多模态学习系统在现实场景中的广泛应用，安全问题变得越来越突出。对于这一领域的安全问题缺乏系统性研究已成为一个重要的障碍。为了解决这个问题，我们提出了第一个多模态学习系统安全的分类法，确定了这些问题的四个关键支柱。借助这一分类法，我们对每个支柱进行了深入审查，突出了当前发展状态的关键限制。最后，我们指出了多模态学习系统安全面临的独特挑战，并提供了未来研究的潜在方向。

    With the wide deployment of multimodal learning systems (MMLS) in real-world scenarios, safety concerns have become increasingly prominent. The absence of systematic research into their safety is a significant barrier to progress in this field. To bridge the gap, we present the first taxonomy for MMLS safety, identifying four essential pillars of these concerns. Leveraging this taxonomy, we conduct in-depth reviews for each pillar, highlighting key limitations based on the current state of development. Finally, we pinpoint unique challenges in MMLS safety and provide potential directions for future research.
    

