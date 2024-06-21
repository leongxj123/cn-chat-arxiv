# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods](https://arxiv.org/abs/2403.20150) | TFB通过解决数据领域覆盖不足、对传统方法的刻板印象以及不一致、不灵活的流程等问题，推动了时间序列预测方法基准比较的最新技术发展。 |
| [^2] | [Towards Measuring and Modeling "Culture" in LLMs: A Survey](https://arxiv.org/abs/2403.15412) | 这项研究调查了39篇最新论文，旨在研究大型语言模型中的文化表达和包容性，发现当前研究未对“文化”进行定义，而是在特定设计的数据集上对模型进行探究，研究了某些“文化”的方面，留下许多未被探究的有趣和重要方面，如语义领域和关于性。 |
| [^3] | [A Question-centric Multi-experts Contrastive Learning Framework for Improving the Accuracy and Interpretability of Deep Sequential Knowledge Tracing Models](https://arxiv.org/abs/2403.07322) | 通过一个问题中心的多专家对比学习框架，提高深度序列知识追踪模型的准确性和可解释性，解决了知识追踪中个体问题信息建模和模型预测结果解释的重要挑战 |
| [^4] | [Evaluating the Performance of ChatGPT for Spam Email Detection](https://arxiv.org/abs/2402.15537) | 该研究评估了ChatGPT在英文和中文电子邮件数据集中用于垃圾邮件检测的性能，并探讨了其在这一领域的潜力。 |
| [^5] | [Evolving AI Collectives to Enhance Human Diversity and Enable Self-Regulation](https://arxiv.org/abs/2402.12590) | 探索人工智能互动的“类社会”属性，以增加奖励并减少风险，展示新兴的去中心化AI集体如何扩大人类多样性范围和降低在线有毒行为风险。 |
| [^6] | [The Male CEO and the Female Assistant: Probing Gender Biases in Text-To-Image Models Through Paired Stereotype Test](https://arxiv.org/abs/2402.11089) | 通过成对刻板印象测试（PST）框架，在文本-图像模型中探究性别偏见，并评估了DALLE-3在性别职业和组织权力方面的偏见。 |
| [^7] | [A Study of Fairness Concerns in AI-based Mobile App Reviews](https://arxiv.org/abs/2401.08097) | 本文研究了AI基于移动应用评价中的公平关注，并通过构建数据集和开发分类器的方法，成功检测出公平性评论，并识别出约92000条公平性评论。 |
| [^8] | [Remembering to Be Fair: Non-Markovian Fairness in Sequential Decision Making](https://arxiv.org/abs/2312.04772) | 本文研究了在序贯决策过程中的非马尔可夫公平性，发现公平往往取决于历史，需要在过程中的不同时间点进行评估。 |
| [^9] | [FRAPP\'E: A Group Fairness Framework for Post-Processing Everything](https://arxiv.org/abs/2312.02592) | 提出了一个将任何正则化的处理中方法转化为后处理方法的框架，适用于更广泛的问题设置，保持了良好的公平错误权衡，并且可能提高之前方法的效力 |
| [^10] | [CDEval: A Benchmark for Measuring the Cultural Dimensions of Large Language Models](https://arxiv.org/abs/2311.16421) | CDEval是一个新的基准，用于评估大规模语言模型的文化维度。通过对六个文化维度和七个领域的全面实验，我们揭示了主流语言模型的文化特征、一致性和差异。这些发现强调了在开发语言模型时融入文化考虑的重要性。 |
| [^11] | [Human Action Co-occurrence in Lifestyle Vlogs using Graph Link Prediction.](http://arxiv.org/abs/2309.06219) | 该论文提出了自动识别人类动作共现的任务，并创建了ACE数据集以及相应的代码。通过利用视觉和文本信息的图链接预测模型，可以有效捕捉不同数据域中的人类动作关系。 |
| [^12] | [The Challenges of Machine Learning for Trust and Safety: A Case Study on Misinformation Detection.](http://arxiv.org/abs/2308.12215) | 本研究通过虚假信息检测为例，检查了机器学习在信任与安全问题中学术与实践之间的脱节，并发现了文献中存在的严重不足之处，包括任务不符合在线服务面临的挑战、数据集和模型评估不真实、评估不独立于模型训练等。在此基础上，提出了评估机器学习应用于信任与安全问题的建议。 |
| [^13] | [Simulating the Integration of Urban Air Mobility into Existing Transportation Systems: A Survey.](http://arxiv.org/abs/2301.12901) | 本文调查了城市空中出行（UAM）在大都市交通中的研究现状，确定了将UAM融入城市交通系统的关键挑战和机遇，包括对现有交通模式和拥堵的影响；安全分析和风险评估；潜在的经济和环境效益；以及为UAM和地面交通开发共享基础设施和路线。同时，我们讨论了UAM的潜在好处，如缩短旅行时间和改善服务不足地区的可达性。 |

# 详细

[^1]: TFB：面向时间序列预测方法全面且公平的基准比较

    TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods

    [https://arxiv.org/abs/2403.20150](https://arxiv.org/abs/2403.20150)

    TFB通过解决数据领域覆盖不足、对传统方法的刻板印象以及不一致、不灵活的流程等问题，推动了时间序列预测方法基准比较的最新技术发展。

    

    时间序列会在经济、交通、健康和能源等不同领域中产生，对未来数值的预测在许多重要应用中起着关键作用。不出所料，许多预测方法被提出。为了确保进展，有必要能够以全面且可靠的方式经验性地研究和比较这些方法。为了实现这一目标，我们提出了TFB，一个自动化的时间序列预测（TSF）方法基准测试。TFB通过解决与数据集、比较方法和评估管道相关的缺点，推动了最新技术的发展：1）数据领域覆盖不足，2）对传统方法的刻板印象，3）不一致和不灵活的流程。为了获得更好的领域覆盖率，我们包括了来自10个不同领域的数据集：交通、电力、能源、环境、自然、经济、股票市场、银行、健康和网络。我们还提供了一个时间序列特性

    arXiv:2403.20150v1 Announce Type: cross  Abstract: Time series are generated in diverse domains such as economic, traffic, health, and energy, where forecasting of future values has numerous important applications. Not surprisingly, many forecasting methods are being proposed. To ensure progress, it is essential to be able to study and compare such methods empirically in a comprehensive and reliable manner. To achieve this, we propose TFB, an automated benchmark for Time Series Forecasting (TSF) methods. TFB advances the state-of-the-art by addressing shortcomings related to datasets, comparison methods, and evaluation pipelines: 1) insufficient coverage of data domains, 2) stereotype bias against traditional methods, and 3) inconsistent and inflexible pipelines. To achieve better domain coverage, we include datasets from 10 different domains: traffic, electricity, energy, the environment, nature, economic, stock markets, banking, health, and the web. We also provide a time series char
    
[^2]: 在LLMs中测量和建模“文化”：一项调查

    Towards Measuring and Modeling "Culture" in LLMs: A Survey

    [https://arxiv.org/abs/2403.15412](https://arxiv.org/abs/2403.15412)

    这项研究调查了39篇最新论文，旨在研究大型语言模型中的文化表达和包容性，发现当前研究未对“文化”进行定义，而是在特定设计的数据集上对模型进行探究，研究了某些“文化”的方面，留下许多未被探究的有趣和重要方面，如语义领域和关于性。

    

    我们呈现了对39篇最新论文的调查，旨在研究大型语言模型中的文化表达和包容性。我们观察到，没有一篇研究定义“文化”，这是一个复杂、多层面的概念；相反，它们在一些特别设计的数据集上对模型进行探究，这些数据集代表了某些“文化”的方面。我们将这些方面称为文化的代理，并将它们组织在人口统计、语义和语言文化交互代理的三个维度上。我们还对采用的探查方法进行了分类。我们的分析表明，只有“文化”的某些方面，如价值观和目标，被研究了，留下了几个其他有趣且重要的方面，特别是大量语义领域和关于性（Hershcovich等人，2022）的未被探究。另外两个关键的空白是目前方法的鲁棒性和情境性的缺乏。基于这些观察结果，

    arXiv:2403.15412v1 Announce Type: cross  Abstract: We present a survey of 39 recent papers that aim to study cultural representation and inclusion in large language models. We observe that none of the studies define "culture," which is a complex, multifaceted concept; instead, they probe the models on some specially designed datasets which represent certain aspects of "culture." We call these aspects the proxies of cultures, and organize them across three dimensions of demographic, semantic and linguistic-cultural interaction proxies. We also categorize the probing methods employed. Our analysis indicates that only certain aspects of "culture," such as values and objectives, have been studied, leaving several other interesting and important facets, especially the multitude of semantic domains (Thompson et al., 2020) and aboutness (Hershcovich et al., 2022), unexplored. Two other crucial gaps are the lack of robustness and situatedness of the current methods. Based on these observations
    
[^3]: 一个问题中心的多专家对比学习框架，用于提高深度序列知识追踪模型的准确性和可解释性

    A Question-centric Multi-experts Contrastive Learning Framework for Improving the Accuracy and Interpretability of Deep Sequential Knowledge Tracing Models

    [https://arxiv.org/abs/2403.07322](https://arxiv.org/abs/2403.07322)

    通过一个问题中心的多专家对比学习框架，提高深度序列知识追踪模型的准确性和可解释性，解决了知识追踪中个体问题信息建模和模型预测结果解释的重要挑战

    

    知识追踪在通过分析学生历史学习过程来预测其未来表现中发挥着至关重要的作用。深度神经网络在解决知识追踪问题方面展现出巨大潜力。然而，将深度学习技术应用于模拟知识追踪过程仍然存在一些重要挑战。第一个挑战在于将问题的个体信息融入建模中。这很关键，因为尽管问题共享相同的知识组件（KC），但学生对同质问题的知识习得可以有显著差异。第二个挑战在于解释现有基于深度学习的知识追踪模型的预测结果。在真实应用中，虽然可能并不需要完全透明和可解释的模型参数，但关键是以老师能理解的方式呈现模型的预测结果。

    arXiv:2403.07322v1 Announce Type: cross  Abstract: Knowledge tracing (KT) plays a crucial role in predicting students' future performance by analyzing their historical learning processes. Deep neural networks (DNNs) have shown great potential in solving the KT problem. However, there still exist some important challenges when applying deep learning techniques to model the KT process. The first challenge lies in taking the individual information of the question into modeling. This is crucial because, despite questions sharing the same knowledge component (KC), students' knowledge acquisition on homogeneous questions can vary significantly. The second challenge lies in interpreting the prediction results from existing deep learning-based KT models. In real-world applications, while it may not be necessary to have complete transparency and interpretability of the model parameters, it is crucial to present the model's prediction results in a manner that teachers find interpretable. This ma
    
[^4]: 评估ChatGPT用于垃圾邮件检测的性能

    Evaluating the Performance of ChatGPT for Spam Email Detection

    [https://arxiv.org/abs/2402.15537](https://arxiv.org/abs/2402.15537)

    该研究评估了ChatGPT在英文和中文电子邮件数据集中用于垃圾邮件检测的性能，并探讨了其在这一领域的潜力。

    

    电子邮件继续是专业和商业领域中至关重要且广泛使用的通信媒介。然而，垃圾邮件的普及给用户带来了重大挑战，扰乱了他们的日常工作并降低了生产率。因此，基于内容准确地识别和过滤垃圾邮件对网络安全至关重要。最近自然语言处理领域的发展，特别是大型语言模型如ChatGPT，在诸如问答和文本生成等任务中表现出色。然而，其在垃圾邮件识别方面的潜力尚未得到充分探索。为了填补这一空白，本研究尝试评估ChatGPT在英文和中文电子邮件数据集中用于垃圾邮件识别的能力。我们利用ChatGPT进行垃圾邮件检测，采用上下文学习，需要提示说明和少量示范。

    arXiv:2402.15537v1 Announce Type: cross  Abstract: Email continues to be a pivotal and extensively utilized communication medium within professional and commercial domains. Nonetheless, the prevalence of spam emails poses a significant challenge for users, disrupting their daily routines and diminishing productivity. Consequently, accurately identifying and filtering spam based on content has become crucial for cybersecurity. Recent advancements in natural language processing, particularly with large language models like ChatGPT, have shown remarkable performance in tasks such as question answering and text generation. However, its potential in spam identification remains underexplored. To fill in the gap, this study attempts to evaluate ChatGPT's capabilities for spam identification in both English and Chinese email datasets. We employ ChatGPT for spam email detection using in-context learning, which requires a prompt instruction and a few demonstrations. We also investigate how the t
    
[^5]: 将AI集体进化，增强人类多样性并实现自我调节

    Evolving AI Collectives to Enhance Human Diversity and Enable Self-Regulation

    [https://arxiv.org/abs/2402.12590](https://arxiv.org/abs/2402.12590)

    探索人工智能互动的“类社会”属性，以增加奖励并减少风险，展示新兴的去中心化AI集体如何扩大人类多样性范围和降低在线有毒行为风险。

    

    大型语言模型根据他人生成的文本来引导其行为。这种能力及其在在线环境中日益普及的趋势预示着它们将有意或无意地“编程”彼此并形成新兴的AI主体性、关系和集体。在这里，我们呼吁研究界探究这些互动人工智能的“类社会”属性，以增加其奖励并减少对人类社会和在线环境健康的风险。我们利用一个简单模型及其输出来说明这种新兴的、去中心化的AI集体如何扩大人类多样性范围并降低在线有毒、反社会行为的风险。最后，我们讨论了AI自我调节的机会，并解决了涉及创建和维护去中心化AI集体的伦理问题和设计挑战。

    arXiv:2402.12590v1 Announce Type: new  Abstract: Large language models steer their behaviors based on texts generated by others. This capacity and their increasing prevalence in online settings portend that they will intentionally or unintentionally "program" one another and form emergent AI subjectivities, relationships, and collectives. Here, we call upon the research community to investigate these "society-like" properties of interacting artificial intelligences to increase their rewards and reduce their risks for human society and the health of online environments. We use a simple model and its outputs to illustrate how such emergent, decentralized AI collectives can expand the bounds of human diversity and reduce the risk of toxic, anti-social behavior online. Finally, we discuss opportunities for AI self-moderation and address ethical issues and design challenges associated with creating and maintaining decentralized AI collectives.
    
[^6]: 男性CEO和女性助理：通过成对刻板印象测试探究文本-图像模型中的性别偏见

    The Male CEO and the Female Assistant: Probing Gender Biases in Text-To-Image Models Through Paired Stereotype Test

    [https://arxiv.org/abs/2402.11089](https://arxiv.org/abs/2402.11089)

    通过成对刻板印象测试（PST）框架，在文本-图像模型中探究性别偏见，并评估了DALLE-3在性别职业和组织权力方面的偏见。

    

    最近大规模的文本到图像（T2I）模型（如DALLE-3）展示了在新应用中的巨大潜力，但也面临前所未有的公平挑战。先前的研究揭示了单人图像生成中的性别偏见，但T2I模型应用可能需要同时描绘两个或更多人。该设定中的潜在偏见仍未被探究，导致使用中的公平相关风险。为了研究T2I模型中性别偏见的基本方面，我们提出了一种新颖的成对刻板印象测试（PST）偏见评估框架。PST促使模型生成同一图像中的两个个体，用与相反性别刻板印象相关联的两个社会身份来描述他们。通过生成的图像遵从性别刻板印象的程度来衡量偏见。利用PST，我们从两个角度评估DALLE-3：性别职业中的偏见和组织权力中的偏见。

    arXiv:2402.11089v1 Announce Type: cross  Abstract: Recent large-scale Text-To-Image (T2I) models such as DALLE-3 demonstrate great potential in new applications, but also face unprecedented fairness challenges. Prior studies revealed gender biases in single-person image generation, but T2I model applications might require portraying two or more people simultaneously. Potential biases in this setting remain unexplored, leading to fairness-related risks in usage. To study these underlying facets of gender biases in T2I models, we propose a novel Paired Stereotype Test (PST) bias evaluation framework. PST prompts the model to generate two individuals in the same image. They are described with two social identities that are stereotypically associated with the opposite gender. Biases can then be measured by the level of conformation to gender stereotypes in generated images. Using PST, we evaluate DALLE-3 from 2 perspectives: biases in gendered occupation and biases in organizational power.
    
[^7]: AI基于移动应用评价的公平关注研究

    A Study of Fairness Concerns in AI-based Mobile App Reviews

    [https://arxiv.org/abs/2401.08097](https://arxiv.org/abs/2401.08097)

    本文研究了AI基于移动应用评价中的公平关注，并通过构建数据集和开发分类器的方法，成功检测出公平性评论，并识别出约92000条公平性评论。

    

    公平是AI系统中必须解决的社会技术问题之一。不公平的AI系统，特别是不公平的AI基于移动应用，可能给全球很大一部分人口带来困难。本文旨在分析AI基于应用评价中的公平问题。我们首先手动构建了一个基准数据集，包括公平性和非公平性评论的统计样本。利用这个基准数据集，我们开发和评估了一组机器学习和深度学习分类器，用于区分公平性评论和非公平性评论。我们的实验结果显示，我们最佳的分类器可以以94%的精确度检测到公平性评论。然后，我们将最佳分类器应用于从108个AI基于应用收集的约950万条评论，识别出约92000条公平性评论。接下来，我们将K-means聚类技术应用于这92000条公平性评论。

    arXiv:2401.08097v2 Announce Type: replace-cross Abstract: Fairness is one of the socio-technical concerns that must be addressed in AI-based systems. Unfair AI-based systems, particularly unfair AI-based mobile apps, can pose difficulties for a significant proportion of the global population. This paper aims to analyze fairness concerns in AI-based app reviews.We first manually constructed a ground-truth dataset, including a statistical sample of fairness and non-fairness reviews. Leveraging the ground-truth dataset, we developed and evaluated a set of machine learning and deep learning classifiers that distinguish fairness reviews from non-fairness reviews. Our experiments show that our best-performing classifier can detect fairness reviews with a precision of 94%. We then applied the best-performing classifier on approximately 9.5M reviews collected from 108 AI-based apps and identified around 92K fairness reviews. Next, applying the K-means clustering technique to the 92K fairness r
    
[^8]: 在序贯决策中记住公平：非马尔可夫公平性

    Remembering to Be Fair: Non-Markovian Fairness in Sequential Decision Making

    [https://arxiv.org/abs/2312.04772](https://arxiv.org/abs/2312.04772)

    本文研究了在序贯决策过程中的非马尔可夫公平性，发现公平往往取决于历史，需要在过程中的不同时间点进行评估。

    

    公平的决策制定在很大程度上是针对单一决策进行研究的。本文在多个利益相关者可能受到决策结果影响的情况下，研究了顺序决策中的公平概念。我们观察到，公平往往取决于顺序决策过程的历史，从这个意义上讲，它是固有的非马尔可夫性。我们进一步观察到，公平通常需要在过程中的某个时间点进行评估，而不仅仅是在过程结束时。为了推进我们对这类公平性问题的理解，我们探讨了顺序决策背景下非马尔可夫公平的概念。我们确定了非马尔可夫公平的属性，包括长期公平性、任意时刻公平性、周期性公平性和有界公平性等概念。我们进一步探讨了非马尔可夫公平性和记忆之间的相互作用，以及这如何支持制定公平政策。

    arXiv:2312.04772v3 Announce Type: replace  Abstract: Fair decision making has largely been studied with respect to a single decision. In this paper we investigate the notion of fairness in the context of sequential decision making where multiple stakeholders can be affected by the outcomes of decisions. We observe that fairness often depends on the history of the sequential decision-making process, and in this sense that it is inherently non-Markovian. We further observe that fairness often needs to be assessed at time points within the process, not just at the end of the process. To advance our understanding of this class of fairness problems, we explore the notion of non-Markovian fairness in the context of sequential decision making. We identify properties of non-Markovian fairness, including notions of long-term, anytime, periodic, and bounded fairness. We further explore the interplay between non-Markovian fairness and memory, and how this can support construction of fair policies
    
[^9]: FRAPP'E：一个用于后处理的群体公平性框架

    FRAPP\'E: A Group Fairness Framework for Post-Processing Everything

    [https://arxiv.org/abs/2312.02592](https://arxiv.org/abs/2312.02592)

    提出了一个将任何正则化的处理中方法转化为后处理方法的框架，适用于更广泛的问题设置，保持了良好的公平错误权衡，并且可能提高之前方法的效力

    

    尽管在处理中实现了有前途的公平错误权衡，但群体公平性的处理方法无法在许多计算资源有限或无法访问预测模型训练管道的实际应用中使用。在这些情况下，后处理是一个可行的替代方法。然而，当前方法专为特定问题设置和公平性定义而设计，因此不如处理中方法广泛适用。在这项工作中，我们提出了一个框架，将任何正则化的处理中方法转化为后处理方法。这一方法提供了一种获得更广泛问题设置的后处理技术的途径，远超过以前的后处理文献。我们理论上并通过大量实验展示，我们的框架保留了处理中实现的良好公平错误权衡，并且能够提高先前方法的效力。

    arXiv:2312.02592v2 Announce Type: replace  Abstract: Despite achieving promising fairness-error trade-offs, in-processing mitigation techniques for group fairness cannot be employed in numerous practical applications with limited computation resources or no access to the training pipeline of the prediction model. In these situations, post-processing is a viable alternative. However, current methods are tailored to specific problem settings and fairness definitions and hence, are not as broadly applicable as in-processing. In this work, we propose a framework that turns any regularized in-processing method into a post-processing approach. This procedure prescribes a way to obtain post-processing techniques for a much broader range of problem settings than the prior post-processing literature. We show theoretically and through extensive experiments that our framework preserves the good fairness-error trade-offs achieved with in-processing and can improve over the effectiveness of prior p
    
[^10]: CDEval：一个用于衡量大规模语言模型文化维度的基准

    CDEval: A Benchmark for Measuring the Cultural Dimensions of Large Language Models

    [https://arxiv.org/abs/2311.16421](https://arxiv.org/abs/2311.16421)

    CDEval是一个新的基准，用于评估大规模语言模型的文化维度。通过对六个文化维度和七个领域的全面实验，我们揭示了主流语言模型的文化特征、一致性和差异。这些发现强调了在开发语言模型时融入文化考虑的重要性。

    

    随着大规模语言模型（LLMs）的扩展显著增强其能力，对确保其负责任和伦理使用的对齐问题越来越受关注。尽管现有的对齐工作主要集中在普世价值观（如HHH原则）上，但文化这一本质上是众多多元化的维度却未得到充分关注。本研究引入了一个新的基准CDEval，旨在评估LLMs的文化维度。CDEval通过结合GPT-4的自动生成和人工验证构建，在七个领域涵盖了六个文化维度。我们全面的实验为主流LLMs的文化提供了有趣的洞察，突出了不同维度和领域之间的一致性和差异。研究结果强调了在LLM的开发中整合文化考虑的重要性，特别是在多元文化环境中的应用。

    As the scaling of Large Language Models (LLMs) has dramatically enhanced their capabilities, there has been a growing focus on the alignment problem to ensure their responsible and ethical use. While existing alignment efforts predominantly concentrate on universal values such as the HHH principle, the aspect of culture, which is inherently pluralistic and diverse, has not received adequate attention. This work introduces a new benchmark, CDEval, aimed at evaluating the cultural dimensions of LLMs. CDEval is constructed by incorporating both GPT-4's automated generation and human verification, covering six cultural dimensions across seven domains. Our comprehensive experiments provide intriguing insights into the culture of mainstream LLMs, highlighting both consistencies and variations across different dimensions and domains. The findings underscore the importance of integrating cultural considerations in LLM development, particularly for applications in diverse cultural settings. Thr
    
[^11]: 使用图链接预测在生活方式vlog中的人类动作共现

    Human Action Co-occurrence in Lifestyle Vlogs using Graph Link Prediction. (arXiv:2309.06219v1 [cs.CV])

    [http://arxiv.org/abs/2309.06219](http://arxiv.org/abs/2309.06219)

    该论文提出了自动识别人类动作共现的任务，并创建了ACE数据集以及相应的代码。通过利用视觉和文本信息的图链接预测模型，可以有效捕捉不同数据域中的人类动作关系。

    

    我们介绍了自动识别人类动作共现的任务，即确定两个人类动作是否可以在同一时间间隔内共现。我们创建并公开了ACE（Action Co-occurrencE）数据集，该数据集由约12k个共现的视觉动作对和它们对应的视频片段组成的大型图形。我们描述了利用视觉和文本信息来自动推断两个动作是否共现的图链接预测模型。我们证明了图形特别适合捕捉人类动作之间的关系，并且所学习的图形表示对于我们的任务是有效的，并且在不同的数据域中捕捉到新颖而相关的信息。本文介绍的ACE数据集和代码可在https://github.com/MichiganNLP/vlog_action_co-occurrence公开获取。

    We introduce the task of automatic human action co-occurrence identification, i.e., determine whether two human actions can co-occur in the same interval of time. We create and make publicly available the ACE (Action Co-occurrencE) dataset, consisting of a large graph of ~12k co-occurring pairs of visual actions and their corresponding video clips. We describe graph link prediction models that leverage visual and textual information to automatically infer if two actions are co-occurring. We show that graphs are particularly well suited to capture relations between human actions, and the learned graph representations are effective for our task and capture novel and relevant information across different data domains. The ACE dataset and the code introduced in this paper are publicly available at https://github.com/MichiganNLP/vlog_action_co-occurrence.
    
[^12]: 机器学习在信任与安全方面的挑战：一个针对虚假信息检测的案例研究

    The Challenges of Machine Learning for Trust and Safety: A Case Study on Misinformation Detection. (arXiv:2308.12215v1 [cs.LG])

    [http://arxiv.org/abs/2308.12215](http://arxiv.org/abs/2308.12215)

    本研究通过虚假信息检测为例，检查了机器学习在信任与安全问题中学术与实践之间的脱节，并发现了文献中存在的严重不足之处，包括任务不符合在线服务面临的挑战、数据集和模型评估不真实、评估不独立于模型训练等。在此基础上，提出了评估机器学习应用于信任与安全问题的建议。

    

    我们使用虚假信息检测作为案例研究，检查了在将机器学习应用于信任与安全问题上学术和实践之间的脱节。我们对该领域中270篇广受引用的论文进行了自动检测虚假信息的文献系统化，并对子集中的论文进行了数据和代码的可用性、设计失误、可复现性和泛化性等方面的研究。我们发现文献中存在严重的不足之处，这对所声称的性能和实用性提出了质疑。检测任务通常与在线服务真正面临的挑战有本质上的区别。数据集和模型评估通常不代表现实世界的情景，而且评估往往不独立于模型训练。数据和代码的可用性很差。模型在领域外的数据上泛化能力不强。基于这些结果，我们提出了评估机器学习应用于信任与安全问题的建议。

    We examine the disconnect between scholarship and practice in applying machine learning to trust and safety problems, using misinformation detection as a case study. We systematize literature on automated detection of misinformation across a corpus of 270 well-cited papers in the field. We then examine subsets of papers for data and code availability, design missteps, reproducibility, and generalizability. We find significant shortcomings in the literature that call into question claimed performance and practicality. Detection tasks are often meaningfully distinct from the challenges that online services actually face. Datasets and model evaluation are often non-representative of real-world contexts, and evaluation frequently is not independent of model training. Data and code availability is poor. Models do not generalize well to out-of-domain data. Based on these results, we offer recommendations for evaluating machine learning applications to trust and safety problems. Our aim is fo
    
[^13]: 模拟城市空中出行融入现有交通系统：一项调查

    Simulating the Integration of Urban Air Mobility into Existing Transportation Systems: A Survey. (arXiv:2301.12901v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2301.12901](http://arxiv.org/abs/2301.12901)

    本文调查了城市空中出行（UAM）在大都市交通中的研究现状，确定了将UAM融入城市交通系统的关键挑战和机遇，包括对现有交通模式和拥堵的影响；安全分析和风险评估；潜在的经济和环境效益；以及为UAM和地面交通开发共享基础设施和路线。同时，我们讨论了UAM的潜在好处，如缩短旅行时间和改善服务不足地区的可达性。

    This paper surveys the current state of research on urban air mobility (UAM) in metropolitan-scale traffic using simulation techniques, identifying key challenges and opportunities for integrating UAM into urban transportation systems, including impacts on existing traffic patterns and congestion, safety analysis and risk assessment, potential economic and environmental benefits, and the development of shared infrastructure and routes for UAM and ground-based transportation. The potential benefits of UAM, such as reduced travel times and improved accessibility for underserved areas, are also discussed.

    城市空中出行（UAM）有可能彻底改变大都市地区的交通方式，提供一种新的交通方式，缓解拥堵，提高可达性。然而，将UAM融入现有交通系统是一项复杂的任务，需要深入了解其对交通流量和容量的影响。在本文中，我们进行了一项调查，使用模拟技术调查了UAM在大都市交通中的研究现状。我们确定了将UAM融入城市交通系统的关键挑战和机遇，包括对现有交通模式和拥堵的影响；安全分析和风险评估；潜在的经济和环境效益；以及为UAM和地面交通开发共享基础设施和路线。我们还讨论了UAM的潜在好处，如缩短旅行时间和改善服务不足地区的可达性。我们的调查

    Urban air mobility (UAM) has the potential to revolutionize transportation in metropolitan areas, providing a new mode of transportation that could alleviate congestion and improve accessibility. However, the integration of UAM into existing transportation systems is a complex task that requires a thorough understanding of its impact on traffic flow and capacity. In this paper, we conduct a survey to investigate the current state of research on UAM in metropolitan-scale traffic using simulation techniques. We identify key challenges and opportunities for the integration of UAM into urban transportation systems, including impacts on existing traffic patterns and congestion; safety analysis and risk assessment; potential economic and environmental benefits; and the development of shared infrastructure and routes for UAM and ground-based transportation. We also discuss the potential benefits of UAM, such as reduced travel times and improved accessibility for underserved areas. Our survey 
    

