# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Toolbox for Surfacing Health Equity Harms and Biases in Large Language Models](https://arxiv.org/abs/2403.12025) | 提出了用于揭示大型语言模型中健康公平危害和偏见的资源和方法，进行了实证案例研究，并提出了用于人类评估LLM生成答案偏见的多因子框架以及EquityMedQA数据集。 |
| [^2] | [Large Language Models are Geographically Biased](https://arxiv.org/abs/2402.02680) | 本文研究了大型语言模型的地理偏见，并展示了其对地理空间预测的系统错误，通过零射击地理空间预测来评估其对世界的认知。 |

# 详细

[^1]: 一个用于揭示大型语言模型中健康公平危害和偏见的工具箱

    A Toolbox for Surfacing Health Equity Harms and Biases in Large Language Models

    [https://arxiv.org/abs/2403.12025](https://arxiv.org/abs/2403.12025)

    提出了用于揭示大型语言模型中健康公平危害和偏见的资源和方法，进行了实证案例研究，并提出了用于人类评估LLM生成答案偏见的多因子框架以及EquityMedQA数据集。

    

    大型语言模型（LLMs）有着为复杂的健康信息需求提供服务的巨大潜力，但同时也有可能引入危害并加剧健康不平等。可靠地评估与公平相关的模型失灵是发展促进健康公平系统的关键步骤。在这项工作中，我们提出了用于揭示可能导致LLM生成的长篇答案中的公平相关危害的偏见的资源和方法，并使用Med-PaLM 2进行了一项实证案例研究，这是迄今为止在该领域进行的最大规模的人类评估研究。我们的贡献包括用于人类评估LLM生成答案偏见的多因子框架，以及EquityMedQA，一个包含七个新发布数据集的收集，其中既包括手动策划又包括LLM生成的问题，丰富了对抗性查询。我们的人类评估框架和数据集设计过程都根植于实际

    arXiv:2403.12025v1 Announce Type: cross  Abstract: Large language models (LLMs) hold immense promise to serve complex health information needs but also have the potential to introduce harm and exacerbate health disparities. Reliably evaluating equity-related model failures is a critical step toward developing systems that promote health equity. In this work, we present resources and methodologies for surfacing biases with potential to precipitate equity-related harms in long-form, LLM-generated answers to medical questions and then conduct an empirical case study with Med-PaLM 2, resulting in the largest human evaluation study in this area to date. Our contributions include a multifactorial framework for human assessment of LLM-generated answers for biases, and EquityMedQA, a collection of seven newly-released datasets comprising both manually-curated and LLM-generated questions enriched for adversarial queries. Both our human assessment framework and dataset design process are grounde
    
[^2]: 大型语言模型存在地理偏见

    Large Language Models are Geographically Biased

    [https://arxiv.org/abs/2402.02680](https://arxiv.org/abs/2402.02680)

    本文研究了大型语言模型的地理偏见，并展示了其对地理空间预测的系统错误，通过零射击地理空间预测来评估其对世界的认知。

    

    大型语言模型（LLMs）内在地含有其训练语料库中的偏见，这可能导致社会伤害的持续存在。随着这些基础模型的影响力不断增长，理解和评估它们的偏见对于实现公正和准确性至关重要。本文提出通过地理视角研究LLMs对我们所生活的世界的认知。这种方法特别强大，因为对人类生活中诸多与地理空间相关的方面（如文化、种族、语言、政治和宗教）有着明显的真实性。我们展示了各种问题地理偏见，我们将其定义为地理空间预测中的系统错误。首先，我们证明LLMs能够进行精确的零射击地理空间预测，以评级的形式呈现，其与真实情况之间呈现出强烈的单调相关性（Spearman's ρ最高可达0.89）。然后，我们展示了LLMs在多个客观和子领域上表现出共同的偏见。

    Large Language Models (LLMs) inherently carry the biases contained in their training corpora, which can lead to the perpetuation of societal harm. As the impact of these foundation models grows, understanding and evaluating their biases becomes crucial to achieving fairness and accuracy. We propose to study what LLMs know about the world we live in through the lens of geography. This approach is particularly powerful as there is ground truth for the numerous aspects of human life that are meaningfully projected onto geographic space such as culture, race, language, politics, and religion. We show various problematic geographic biases, which we define as systemic errors in geospatial predictions. Initially, we demonstrate that LLMs are capable of making accurate zero-shot geospatial predictions in the form of ratings that show strong monotonic correlation with ground truth (Spearman's $\rho$ of up to 0.89). We then show that LLMs exhibit common biases across a range of objective and sub
    

