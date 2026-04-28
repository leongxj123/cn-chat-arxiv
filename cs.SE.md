# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Automated Approaches to Detect Self-Admitted Technical Debt: A Systematic Literature Review](https://arxiv.org/abs/2312.15020) | 论文提出了一种特征提取技术和ML/DL算法分类法，旨在比较和基准测试其在技术债务检测中的表现。 |
| [^2] | [Automatically Estimating the Effort Required to Repay Self-Admitted Technical Debt.](http://arxiv.org/abs/2309.06020) | 本研究提出了一种新的方法，利用大规模的数据集自动估算自认技术债务的还款工作量。研究结果表明，不同类型的自认技术债务需要不同程度的还款工作量。 |

# 详细

[^1]: 自动化方法检测自我承认的技术债务：系统文献综述

    Automated Approaches to Detect Self-Admitted Technical Debt: A Systematic Literature Review

    [https://arxiv.org/abs/2312.15020](https://arxiv.org/abs/2312.15020)

    论文提出了一种特征提取技术和ML/DL算法分类法，旨在比较和基准测试其在技术债务检测中的表现。

    

    技术债务是软件开发中普遍存在的问题，通常源自开发过程中做出的权衡，在影响软件可维护性和阻碍未来开发工作方面起到作用。自我承认的技术债务（SATD）指的是开发人员明确承认代码库中存在的代码质量或设计缺陷。自动检测SATD已经成为一个重要的研究领域，旨在帮助开发人员高效地识别和解决技术债务。然而，文献中广泛采用的NLP特征提取方法和算法种类多样化常常阻碍研究人员试图提高其性能。基于此，本系统文献综述提出了一种特征提取技术和ML/DL算法分类法，其目的是比较和基准测试所考察研究中它们的性能。我们选择......

    arXiv:2312.15020v2 Announce Type: replace-cross  Abstract: Technical debt is a pervasive issue in software development, often arising from trade-offs made during development, which can impede software maintainability and hinder future development efforts. Self-admitted technical debt (SATD) refers to instances where developers explicitly acknowledge suboptimal code quality or design flaws in the codebase. Automated detection of SATD has emerged as a critical area of research, aiming to assist developers in identifying and addressing technical debt efficiently. However, the enormous variety of feature extraction approaches of NLP and algorithms employed in the literature often hinder researchers from trying to improve their performance. In light of this, this systematic literature review proposes a taxonomy of feature extraction techniques and ML/DL algorithms used in technical debt detection: its objective is to compare and benchmark their performance in the examined studies. We select
    
[^2]: 自动评估偿还自认技术债务所需的工作量

    Automatically Estimating the Effort Required to Repay Self-Admitted Technical Debt. (arXiv:2309.06020v1 [cs.SE])

    [http://arxiv.org/abs/2309.06020](http://arxiv.org/abs/2309.06020)

    本研究提出了一种新的方法，利用大规模的数据集自动估算自认技术债务的还款工作量。研究结果表明，不同类型的自认技术债务需要不同程度的还款工作量。

    

    技术债务是指在软件开发过程中为了短期利益而做出的次优决策所带来的后果。自认技术债务(SATD)是一种特定形式的技术债务，开发人员明确地在软件的源代码注释和提交消息中记录下来。由于SATD可能阻碍软件的开发和维护，因此有效地解决和优先处理它非常重要。然而，目前的方法缺乏根据SATD的文本描述自动评估其还款工作量的能力。为了解决这个限制，我们提出了一种新的方法，利用一个包括1,060个Apache代码库中共2,568,728个提交的341,740个SATD项目的全面数据集来自动估算SATD还款工作量。我们的研究结果表明，不同类型的SATD需要不同程度的还款工作量，其中代码/设计、需求和测试债务需要更多的工作量。

    Technical debt refers to the consequences of sub-optimal decisions made during software development that prioritize short-term benefits over long-term maintainability. Self-Admitted Technical Debt (SATD) is a specific form of technical debt, explicitly documented by developers within software artifacts such as source code comments and commit messages. As SATD can hinder software development and maintenance, it is crucial to address and prioritize it effectively. However, current methodologies lack the ability to automatically estimate the repayment effort of SATD based on its textual descriptions. To address this limitation, we propose a novel approach for automatically estimating SATD repayment effort, utilizing a comprehensive dataset comprising 341,740 SATD items from 2,568,728 commits across 1,060 Apache repositories. Our findings show that different types of SATD require varying levels of repayment effort, with code/design, requirement, and test debt demanding greater effort compa
    

