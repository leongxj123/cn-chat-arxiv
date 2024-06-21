# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generation of Asset Administration Shell with Large Language Model Agents: Interoperability in Digital Twins with Semantic Node](https://arxiv.org/abs/2403.17209) | 通过大型语言模型代理生成AAS实例模型，实现了在数字孪生中的互操作性，降低了手动创建成本和时间。 |
| [^2] | [A Study of Fairness Concerns in AI-based Mobile App Reviews](https://arxiv.org/abs/2401.08097) | 本文研究了AI基于移动应用评价中的公平关注，并通过构建数据集和开发分类器的方法，成功检测出公平性评论，并识别出约92000条公平性评论。 |
| [^3] | [Towards Enhancing the Reproducibility of Deep Learning Bugs: An Empirical Study.](http://arxiv.org/abs/2401.03069) | 本研究旨在提高深度学习Bug的可复现性，通过构建数据集和确定编辑动作和有用信息，这能够解决目前研究中忽视的问题。 |
| [^4] | [Can GPT-4 Replicate Empirical Software Engineering Research?.](http://arxiv.org/abs/2310.01727) | 本论文研究了GPT-4在新数据上执行实证软件工程研究复制的能力，探讨了其揭示实证软件工程研究方法中假设的潜力和应用前景。 |

# 详细

[^1]: 利用大型语言模型代理生成资产管理外壳：数字孪生和语义节点中的互操作性

    Generation of Asset Administration Shell with Large Language Model Agents: Interoperability in Digital Twins with Semantic Node

    [https://arxiv.org/abs/2403.17209](https://arxiv.org/abs/2403.17209)

    通过大型语言模型代理生成AAS实例模型，实现了在数字孪生中的互操作性，降低了手动创建成本和时间。

    

    这项研究介绍了一种新颖的方法，用于协助在工业4.0背景下为数字孪生建模创建资产管理外壳（AAS）实例，旨在增强智能制造中的互操作性，减少手动工作。我们构建了一个“语义节点”数据结构来捕捉文本数据的语义要义。然后，设计并实现了一个由大型语言模型驱动的系统，用于处理“语义节点”并从文本技术数据生成AAS实例模型。我们的评估表明，有效生成率为62-79%，表明相当比例的手动创建工作可以转换为更容易的验证工作，从而减少创建AAS实例模型的时间和成本。在我们的评估中，对不同LLM的比较分析以及检索增强生成（RAG）机制的深入消融研究提供了有关LLM有效性的见解。

    arXiv:2403.17209v1 Announce Type: new  Abstract: This research introduces a novel approach for assisting the creation of Asset Administration Shell (AAS) instances for digital twin modeling within the context of Industry 4.0, aiming to enhance interoperability in smart manufacturing and reduce manual effort. We construct a "semantic node" data structure to capture the semantic essence of textual data. Then, a system powered by large language models is designed and implemented to process "semantic node" and generate AAS instance models from textual technical data. Our evaluation demonstrates a 62-79% effective generation rate, indicating a substantial proportion of manual creation effort can be converted into easier validation effort, thereby reducing the time and cost in creating AAS instance models. In our evaluation, a comparative analysis of different LLMs and an in-depth ablation study of Retrieval-Augmented Generation (RAG) mechanisms provide insights into the effectiveness of LLM
    
[^2]: AI基于移动应用评价的公平关注研究

    A Study of Fairness Concerns in AI-based Mobile App Reviews

    [https://arxiv.org/abs/2401.08097](https://arxiv.org/abs/2401.08097)

    本文研究了AI基于移动应用评价中的公平关注，并通过构建数据集和开发分类器的方法，成功检测出公平性评论，并识别出约92000条公平性评论。

    

    公平是AI系统中必须解决的社会技术问题之一。不公平的AI系统，特别是不公平的AI基于移动应用，可能给全球很大一部分人口带来困难。本文旨在分析AI基于应用评价中的公平问题。我们首先手动构建了一个基准数据集，包括公平性和非公平性评论的统计样本。利用这个基准数据集，我们开发和评估了一组机器学习和深度学习分类器，用于区分公平性评论和非公平性评论。我们的实验结果显示，我们最佳的分类器可以以94%的精确度检测到公平性评论。然后，我们将最佳分类器应用于从108个AI基于应用收集的约950万条评论，识别出约92000条公平性评论。接下来，我们将K-means聚类技术应用于这92000条公平性评论。

    arXiv:2401.08097v2 Announce Type: replace-cross Abstract: Fairness is one of the socio-technical concerns that must be addressed in AI-based systems. Unfair AI-based systems, particularly unfair AI-based mobile apps, can pose difficulties for a significant proportion of the global population. This paper aims to analyze fairness concerns in AI-based app reviews.We first manually constructed a ground-truth dataset, including a statistical sample of fairness and non-fairness reviews. Leveraging the ground-truth dataset, we developed and evaluated a set of machine learning and deep learning classifiers that distinguish fairness reviews from non-fairness reviews. Our experiments show that our best-performing classifier can detect fairness reviews with a precision of 94%. We then applied the best-performing classifier on approximately 9.5M reviews collected from 108 AI-based apps and identified around 92K fairness reviews. Next, applying the K-means clustering technique to the 92K fairness r
    
[^3]: 提高深度学习Bug可复现性的探索性研究

    Towards Enhancing the Reproducibility of Deep Learning Bugs: An Empirical Study. (arXiv:2401.03069v1 [cs.SE])

    [http://arxiv.org/abs/2401.03069](http://arxiv.org/abs/2401.03069)

    本研究旨在提高深度学习Bug的可复现性，通过构建数据集和确定编辑动作和有用信息，这能够解决目前研究中忽视的问题。

    

    背景：深度学习在各个领域取得了显著进展。然而，与传统软件系统一样，深度学习系统也存在Bug，这可能对自动驾驶等领域产生严重影响。尽管深度学习技术取得了重大进展，但很少有研究关注深度学习Bug的可复现性，这阻碍了Bug的解决。现有文献指出，仅有3%的深度学习Bug是可复现的，这凸显了进一步研究的必要性。目标：本文考察深度学习Bug的可复现性，识别可提高深度学习Bug可复现性的编辑动作和有用信息。方法：首先，构建了一个包含来自Stack Overflow和Defects4ML的3个框架和22个架构的668个深度学习Bug的数据集。其次，使用分层抽样选择了102个Bug，并尝试确定它们的可复现性。在复现这些Bug的过程中，我们识别了编辑动作和有用信息。

    Context: Deep learning has achieved remarkable progress in various domains. However, like traditional software systems, deep learning systems contain bugs, which can have severe impacts, as evidenced by crashes involving autonomous vehicles. Despite substantial advancements in deep learning techniques, little research has focused on reproducing deep learning bugs, which hinders resolving them. Existing literature suggests that only 3% of deep learning bugs are reproducible, underscoring the need for further research.  Objective: This paper examines the reproducibility of deep learning bugs. We identify edit actions and useful information that could improve deep learning bug reproducibility.  Method: First, we construct a dataset of 668 deep learning bugs from Stack Overflow and Defects4ML across 3 frameworks and 22 architectures. Second, we select 102 bugs using stratified sampling and try to determine their reproducibility. While reproducing these bugs, we identify edit actions and us
    
[^4]: GPT-4能否复制实证软件工程研究？

    Can GPT-4 Replicate Empirical Software Engineering Research?. (arXiv:2310.01727v1 [cs.SE])

    [http://arxiv.org/abs/2310.01727](http://arxiv.org/abs/2310.01727)

    本论文研究了GPT-4在新数据上执行实证软件工程研究复制的能力，探讨了其揭示实证软件工程研究方法中假设的潜力和应用前景。

    

    对生产系统的实证软件工程研究为从业者和研究人员带来了对软件工程过程的更好理解。然而，只有生产系统的一小部分进行了研究，限制了该研究的影响力。虽然软件工程从业者可以通过复制研究来获得充实自己数据的好处，但这也带来了一系列挑战，因为执行复制需要对研究方法和软件工程数据中的微妙细节有深入的理解。鉴于大型语言模型（LLMs）如GPT-4在处理软件工程和科学任务方面显示出潜力，这些模型可以帮助推广实证软件工程研究。在这篇论文中，我们研究了LLMs在新数据上执行实证软件工程研究复制的能力。我们特别研究了它们揭示实证软件工程研究方法中所做的假设的能力。

    Empirical software engineering research on production systems has brought forth a better understanding of the software engineering process for practitioners and researchers alike. However, only a small subset of production systems is studied, limiting the impact of this research. While software engineering practitioners benefit from replicating research on their own data, this poses its own set of challenges, since performing replications requires a deep understanding of research methodologies and subtle nuances in software engineering data. Given that large language models (LLMs), such as GPT-4, show promise in tackling both software engineering- and science-related tasks, these models could help democratize empirical software engineering research.  In this paper, we examine LLMs' abilities to perform replications of empirical software engineering research on new data. We specifically study their ability to surface assumptions made in empirical software engineering research methodolog
    

