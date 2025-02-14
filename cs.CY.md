# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AI Oversight and Human Mistakes: Evidence from Centre Court.](http://arxiv.org/abs/2401.16754) | 人工智能系统在纠正人类错误方面起到了积极作用，但此举也潜在导致心理成本，并影响人的决策。通过研究网球比赛中的Hawk-Eye审查系统，我们发现引入AI监督后，裁判员的错误率下降，心理成本导致他们更倾向于将球判为进界，从而产生了类型错判的转变。 |
| [^2] | [Curating corpora with classifiers: A case study of clean energy sentiment online.](http://arxiv.org/abs/2305.03092) | 本文介绍了利用分类器来快速选择最佳的相关文档语料库进行分析的方法，探索了过滤掉不相关的推文的方法，以进行在线清洁能源情感分析。 |
| [^3] | [On the Creativity of Large Language Models.](http://arxiv.org/abs/2304.00008) | 这篇论文探讨了大型语言模型的创造性问题，分析了与之相关的机器创造性的难点和易点，并重点分析了这些技术在创意产业中的社会影响。 |

# 详细

[^1]: AI监督和人类错误：来自中心法庭的证据

    AI Oversight and Human Mistakes: Evidence from Centre Court. (arXiv:2401.16754v1 [cs.LG])

    [http://arxiv.org/abs/2401.16754](http://arxiv.org/abs/2401.16754)

    人工智能系统在纠正人类错误方面起到了积极作用，但此举也潜在导致心理成本，并影响人的决策。通过研究网球比赛中的Hawk-Eye审查系统，我们发现引入AI监督后，裁判员的错误率下降，心理成本导致他们更倾向于将球判为进界，从而产生了类型错判的转变。

    

    在机器学习算法不断提升的驱动下，人工智能（AI）系统已经开始在许多场合用于纠正人类错误。我们提供了首个实地证据，证明这种AI监督会产生心理成本，影响人的决策。我们调查了AI监督发生的最高可见性场景之一：顶级网球比赛中裁判的Hawk-Eye审查。我们发现，引入Hawk-Eye审查后，裁判的整体错误率降低，符合心理成本被AI否定的合理忽视现象。我们还发现，裁判增加了对球入内的判定率，从而产生了从II类错误（将球判为出界，实际上是进界）到I类错误（将球判为进界，实际上是出界）的转变。通过对理性不注意的裁判模型进行心理成本的结构估计，我们的结果表明，由于AI否定的心理成本，裁判员降低了错误判定的风险并提高了球入内的判定率。

    Powered by the increasing predictive capabilities of machine learning algorithms, artificial intelligence (AI) systems have begun to be used to overrule human mistakes in many settings. We provide the first field evidence this AI oversight carries psychological costs that can impact human decision-making. We investigate one of the highest visibility settings in which AI oversight has occurred: the Hawk-Eye review of umpires in top tennis tournaments. We find that umpires lowered their overall mistake rate after the introduction of Hawk-Eye review, in line with rational inattention given psychological costs of being overruled by AI. We also find that umpires increased the rate at which they called balls in, which produced a shift from making Type II errors (calling a ball out when in) to Type I errors (calling a ball in when out). We structurally estimate the psychological costs of being overruled by AI using a model of rational inattentive umpires, and our results suggest that because 
    
[^2]: 利用分类器来筛选语料库：以在线清洁能源情感分析为例

    Curating corpora with classifiers: A case study of clean energy sentiment online. (arXiv:2305.03092v1 [cs.CL])

    [http://arxiv.org/abs/2305.03092](http://arxiv.org/abs/2305.03092)

    本文介绍了利用分类器来快速选择最佳的相关文档语料库进行分析的方法，探索了过滤掉不相关的推文的方法，以进行在线清洁能源情感分析。

    

    精心策划的、大规模的社交媒体帖子语料库是补充传统调查的替代数据来源，可以提供广泛的公众意见。虽然调查在收集代表性样本和实现高准确率方面很有效，但运行成本很高，而且会滞后于公众意见数天或数周。这两个缺点可以通过实时、高容量的数据流和快速的分析管道克服。在组织这样的数据管道方面的一个核心挑战是设计一种有效的方法，快速选择最佳的相关文档语料库进行分析。仅仅通过关键词查询往往会包括不相关的文档，而这些文档很难用词袋自然语言处理方法消歧。在这里，我们使用预先训练的基于转换器的模型，通过在手动标注的推文上对其进行微调，探索了语料库策划的方法，以过滤掉不相关的推文。我们能够实现高达0.8以上的F1得分。

    Well curated, large-scale corpora of social media posts containing broad public opinion offer an alternative data source to complement traditional surveys. While surveys are effective at collecting representative samples and are capable of achieving high accuracy, they can be both expensive to run and lag public opinion by days or weeks. Both of these drawbacks could be overcome with a real-time, high volume data stream and fast analysis pipeline. A central challenge in orchestrating such a data pipeline is devising an effective method for rapidly selecting the best corpus of relevant documents for analysis. Querying with keywords alone often includes irrelevant documents that are not easily disambiguated with bag-of-words natural language processing methods. Here, we explore methods of corpus curation to filter irrelevant tweets using pre-trained transformer-based models, fine-tuned for our binary classification task on hand-labeled tweets. We are able to achieve F1 scores of up to 0.
    
[^3]: 关于大型语言模型的创造性研究

    On the Creativity of Large Language Models. (arXiv:2304.00008v1 [cs.AI])

    [http://arxiv.org/abs/2304.00008](http://arxiv.org/abs/2304.00008)

    这篇论文探讨了大型语言模型的创造性问题，分析了与之相关的机器创造性的难点和易点，并重点分析了这些技术在创意产业中的社会影响。

    

    大型语言模型(LLMs)正在颠覆人工智能的多个领域。其中最显著的应用之一是创作，例如诗歌或故事：生成的输出通常具有惊人的质量。但是，一个自然的问题是：LLMs真的可以被认为是创造性的吗？在本文中，我们首先通过创造性理论的角度分析了LLMs的发展，探讨了关键的未解决问题和挑战。然后，我们在与LLMs相关的机器创造性方面确定了一组“易”和“难”问题，并对其进行了讨论。最后，我们分析了这些技术在创意产业中的社会影响。

    Large Language Models (LLMs) are revolutionizing several areas of Artificial Intelligence. One of the most remarkable applications is creative writing, e.g., poetry or storytelling: the generated outputs are often of astonishing quality. However, a natural question arise: can LLMs really be considered creative? In this article we firstly analyze the development of LLMs under the lens of creativity theories, investigating the key open questions and challenges. Then, we identify a set of "easy" and "hard" problems in machine creativity, discussing them in relation to LLMs. Finally, we analyze the societal impact of these technologies with a particular focus on the creative industries.
    

