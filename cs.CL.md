# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [BAT: Learning to Reason about Spatial Sounds with Large Language Models](https://rss.arxiv.org/abs/2402.01591) | 本文提出了BAT，它结合了双耳声音场景分析模型的空间声音感知能力和大规模语言模型的自然语言推理能力，以复制人类的空间声音推理能力。通过使用合成的双耳音频数据集和基于空间声音的问答数据集进行训练，BAT在空间声音感知和推理方面取得了强大的性能。 |
| [^2] | [Leveraging Large Language Models for Analyzing Blood Pressure Variations Across Biological Sex from Scientific Literature](https://arxiv.org/abs/2402.01826) | 本研究利用大型语言模型GPT-35-turbo，从PubMed的25 million个摘要中自动提取出男性和女性的血压平均值和标准差，填补了在考虑生物学性别差异的血压测量方差方面的研究空白。 |

# 详细

[^1]: BAT: 使用大规模语言模型学习关于空间声音的推理能力

    BAT: Learning to Reason about Spatial Sounds with Large Language Models

    [https://rss.arxiv.org/abs/2402.01591](https://rss.arxiv.org/abs/2402.01591)

    本文提出了BAT，它结合了双耳声音场景分析模型的空间声音感知能力和大规模语言模型的自然语言推理能力，以复制人类的空间声音推理能力。通过使用合成的双耳音频数据集和基于空间声音的问答数据集进行训练，BAT在空间声音感知和推理方面取得了强大的性能。

    

    空间声音推理是一种基本的人类技能，它使我们能够根据声音来导航和解释我们的周围环境。本文提出了BAT，它将双耳声音场景分析模型的空间声音感知能力与大规模语言模型（LLM）的自然语言推理能力相结合，以复制这种固有能力。为了解决现有野外空间声音数据集的缺乏，我们使用AudioSet和SoundSpaces 2.0合成了一个双耳音频数据集。接下来，我们开发了一种基于空间声音的问答数据集SpatialSoundQA，提供了一系列QA任务，以训练BAT在空间声音感知和推理的各个方面。BAT的声学前端编码器是一种名为Spatial Audio Spectrogram Transformer（Spatial-AST）的创新空间音频编码器，它本身在声音事件检测、空间定位和距离估计等方面具有强大的性能。通过将Spatial-AST与LLaMA-2 7B集成，

    Spatial sound reasoning is a fundamental human skill, enabling us to navigate and interpret our surroundings based on sound. In this paper we present BAT, which combines the spatial sound perception ability of a binaural acoustic scene analysis model with the natural language reasoning capabilities of a large language model (LLM) to replicate this innate ability. To address the lack of existing datasets of in-the-wild spatial sounds, we synthesized a binaural audio dataset using AudioSet and SoundSpaces 2.0. Next, we developed SpatialSoundQA, a spatial sound-based question-answering dataset, offering a range of QA tasks that train BAT in various aspects of spatial sound perception and reasoning. The acoustic front end encoder of BAT is a novel spatial audio encoder named Spatial Audio Spectrogram Transformer, or Spatial-AST, which by itself achieves strong performance across sound event detection, spatial localization, and distance estimation. By integrating Spatial-AST with LLaMA-2 7B
    
[^2]: 利用大型语言模型分析科学文献中血压变异的生物学性别差异

    Leveraging Large Language Models for Analyzing Blood Pressure Variations Across Biological Sex from Scientific Literature

    [https://arxiv.org/abs/2402.01826](https://arxiv.org/abs/2402.01826)

    本研究利用大型语言模型GPT-35-turbo，从PubMed的25 million个摘要中自动提取出男性和女性的血压平均值和标准差，填补了在考虑生物学性别差异的血压测量方差方面的研究空白。

    

    高血压被定义为高于正常的血压，它在公共卫生领域具有至关重要的意义，因为它是各种心血管疾病的重要先兆，并且显著 contributed to the elevated mortality rates worldwide。然而，许多现有的血压测量技术和标准可能存在偏差，因为它们未考虑临床结果、合并症或人口统计因素，使其在诊断目的上没有结论性。针对这些变量，在研究血压测量方差方面，有限的以数据驱动的研究。在本研究中，我们采用了GPT-35-turbo，一种大型语言模型（LLM），从PubMed的2500万个摘要的数据集中自动提取了男性和女性的血压平均值和标准差数值。符合我们预定义的纳入标准的摘要文章有993篇（即具有与血压、血压单位（如mmHg）和提及相关性的参考文献）。

    Hypertension, defined as blood pressure (BP) that is above normal, holds paramount significance in the realm of public health, as it serves as a critical precursor to various cardiovascular diseases (CVDs) and significantly contributes to elevated mortality rates worldwide. However, many existing BP measurement technologies and standards might be biased because they do not consider clinical outcomes, comorbidities, or demographic factors, making them inconclusive for diagnostic purposes. There is limited data-driven research focused on studying the variance in BP measurements across these variables. In this work, we employed GPT-35-turbo, a large language model (LLM), to automatically extract the mean and standard deviation values of BP for both males and females from a dataset comprising 25 million abstracts sourced from PubMed. 993 article abstracts met our predefined inclusion criteria (i.e., presence of references to blood pressure, units of blood pressure such as mmHg, and mention
    

