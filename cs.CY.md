# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rehabilitation Exercise Quality Assessment through Supervised Contrastive Learning with Hard and Soft Negatives](https://arxiv.org/abs/2403.02772) | 这项研究引入了一种新的有监督对比学习框架，通过结合硬和软负样本，有效地解决了康复锻炼评估中样本稀缺的问题 |
| [^2] | [Beyond prompt brittleness: Evaluating the reliability and consistency of political worldviews in LLMs](https://arxiv.org/abs/2402.17649) | 该研究评估了大型语言模型中的政治世界观的可靠性和一致性，发现他们的可靠性随模型参数数量增加而增加，且在政策方案上有所不同。 |
| [^3] | [Casteist but Not Racist? Quantifying Disparities in Large Language Model Bias between India and the West.](http://arxiv.org/abs/2309.08573) | 本研究量化了大型语言模型在印度和西方上的陈规偏见差异，并开发了一个新的数据集来评估种姓和宗教上的刻板印象。研究发现大多数测试的模型在印度背景下对刻板印象有显著偏见，尤其是与西方背景相比。此外，研究探索了一种简单干预方法来减轻这种偏见的效果。 |

# 详细

[^1]: 通过有监督对比学习进行康复锻炼质量评估，结合硬负样本和软负样本

    Rehabilitation Exercise Quality Assessment through Supervised Contrastive Learning with Hard and Soft Negatives

    [https://arxiv.org/abs/2403.02772](https://arxiv.org/abs/2403.02772)

    这项研究引入了一种新的有监督对比学习框架，通过结合硬和软负样本，有效地解决了康复锻炼评估中样本稀缺的问题

    

    基于锻炼的康复计划已被证明在提高生活质量、降低死亡率和再住院率方面是有效的。利用人工智能驱动的虚拟康复，患者可以在家独立完成锻炼，利用AI算法分析锻炼数据，为患者提供反馈，并向临床医生更新他们的进展情况。这些计划通常会指定各种锻炼类型，这导致康复锻炼评估数据集面临独特挑战：虽然在整体训练样本中丰富，但这些数据集通常对每种具体锻练类型的样本数量有限。这种差异影响了现有方法训练具有小样本量的每种锻练的可泛化模型的能力。为了解决这个问题，我们的论文引入了一种新颖的带有硬负样本和软负样本的有监督对比学习框架，有效利用了整个

    arXiv:2403.02772v1 Announce Type: cross  Abstract: Exercise-based rehabilitation programs have proven to be effective in enhancing the quality of life and reducing mortality and rehospitalization rates. AI-driven virtual rehabilitation, which allows patients to independently complete exercises at home, utilizes AI algorithms to analyze exercise data, providing feedback to patients and updating clinicians on their progress. These programs commonly prescribe a variety of exercise types, leading to a distinct challenge in rehabilitation exercise assessment datasets: while abundant in overall training samples, these datasets often have a limited number of samples for each individual exercise type. This disparity hampers the ability of existing approaches to train generalizable models with such a small sample size per exercise. Addressing this issue, our paper introduces a novel supervised contrastive learning framework with hard and soft negative samples that effectively utilizes the entir
    
[^2]: 超越提示脆弱性：评估LLMs中政治世界观的可靠性和一致性

    Beyond prompt brittleness: Evaluating the reliability and consistency of political worldviews in LLMs

    [https://arxiv.org/abs/2402.17649](https://arxiv.org/abs/2402.17649)

    该研究评估了大型语言模型中的政治世界观的可靠性和一致性，发现他们的可靠性随模型参数数量增加而增加，且在政策方案上有所不同。

    

    由于大型语言模型（LLMs）在广泛系统中的使用，我们需要了解它们是否嵌入了特定的世界观以及这些观点所反映的内容。最近的研究报告称，当用政治问卷进行提示时，LLMs表现出左倾自由倾向。然而，目前尚不清楚这些倾向是否可靠（对提示变化稳健）以及这种倾向是否在政策和政治倾向上保持一致。我们提出了一系列测试，评估了基于收集自七个欧盟国家的选举建议问卷并标注为政策领域的数据集上LLMs在政治声明上立场的可靠性和一致性。我们研究了参数从7B到70B的LLMs，并发现它们的可靠性随参数数量增加而增加。更大的模型显示总体上与左倾政党更强的一致性，但在政策方案中有所不同：它们表现出（左倾）积极的立场

    arXiv:2402.17649v1 Announce Type: new  Abstract: Due to the widespread use of large language models (LLMs) in ubiquitous systems, we need to understand whether they embed a specific worldview and what these views reflect. Recent studies report that, prompted with political questionnaires, LLMs show left-liberal leanings. However, it is as yet unclear whether these leanings are reliable (robust to prompt variations) and whether the leaning is consistent across policies and political leaning. We propose a series of tests which assess the reliability and consistency of LLMs' stances on political statements based on a dataset of voting-advice questionnaires collected from seven EU countries and annotated for policy domains. We study LLMs ranging in size from 7B to 70B parameters and find that their reliability increases with parameter count. Larger models show overall stronger alignment with left-leaning parties but differ among policy programs: They evince a (left-wing) positive stance to
    
[^3]: 印度也存在种姓主义但不存在种族主义吗？量化印度和西方大型语言模型偏见的差异

    Casteist but Not Racist? Quantifying Disparities in Large Language Model Bias between India and the West. (arXiv:2309.08573v1 [cs.CL])

    [http://arxiv.org/abs/2309.08573](http://arxiv.org/abs/2309.08573)

    本研究量化了大型语言模型在印度和西方上的陈规偏见差异，并开发了一个新的数据集来评估种姓和宗教上的刻板印象。研究发现大多数测试的模型在印度背景下对刻板印象有显著偏见，尤其是与西方背景相比。此外，研究探索了一种简单干预方法来减轻这种偏见的效果。

    

    大型语言模型（LLMs）现在每天被数百万用户使用，他们能够传达社会偏见，使用户遭受再现伤害。已有大量的关于LLM偏见的学术研究存在，但主要采用西方中心视角，相对较少关注全球南方地区的偏见水平和潜在伤害。在本文中，我们量化流行LLMs中的陈规偏见，采用以印度为中心的框架，并比较印度和西方背景下的偏见水平。为此，我们开发了一个新颖的数据集，称为Indian-BhED（印度偏见评估数据集），其中包含种姓和宗教上的刻板和反刻板的例子。我们发现，在印度背景下，大多数测试的LLMs对刻板印象有强烈偏见，尤其是与西方背景相比。最后，我们研究了Instruction Prompting作为一种简单的干预手段来减轻这种偏见，并发现它显著减少了刻板印象和反刻板印象。

    Large Language Models (LLMs), now used daily by millions of users, can encode societal biases, exposing their users to representational harms. A large body of scholarship on LLM bias exists but it predominantly adopts a Western-centric frame and attends comparatively less to bias levels and potential harms in the Global South. In this paper, we quantify stereotypical bias in popular LLMs according to an Indian-centric frame and compare bias levels between the Indian and Western contexts. To do this, we develop a novel dataset which we call Indian-BhED (Indian Bias Evaluation Dataset), containing stereotypical and anti-stereotypical examples for caste and religion contexts. We find that the majority of LLMs tested are strongly biased towards stereotypes in the Indian context, especially as compared to the Western context. We finally investigate Instruction Prompting as a simple intervention to mitigate such bias and find that it significantly reduces both stereotypical and anti-stereoty
    

