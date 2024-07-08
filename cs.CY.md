# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [From Representational Harms to Quality-of-Service Harms: A Case Study on Llama 2 Safety Safeguards](https://arxiv.org/abs/2403.13213) | 本文探讨了针对表现性伤害和服务质量伤害的羊驼2安全保障措施的有效性，并指出了大型语言模型在实用性和安全性之间的权衡关系。 |
| [^2] | [A Question-centric Multi-experts Contrastive Learning Framework for Improving the Accuracy and Interpretability of Deep Sequential Knowledge Tracing Models](https://arxiv.org/abs/2403.07322) | 通过一个问题中心的多专家对比学习框架，提高深度序列知识追踪模型的准确性和可解释性，解决了知识追踪中个体问题信息建模和模型预测结果解释的重要挑战 |
| [^3] | [Improving Low-Resource Knowledge Tracing Tasks by Supervised Pre-training and Importance Mechanism Fine-tuning](https://arxiv.org/abs/2403.06725) | 本文提出了名为LoReKT的低资源知识追踪框架，通过监督预训练和微调重要性机制，旨在从丰富资源的KT数据集中学习可转移的参数和表示来改进低资源知识追踪任务。 |
| [^4] | [How AI Ideas Affect the Creativity, Diversity, and Evolution of Human Ideas: Evidence From a Large, Dynamic Experiment.](http://arxiv.org/abs/2401.13481) | AI思想对个体创造力没有影响，但增加了整体思想多样性的数量和变化速率。 |
| [^5] | [Characteristics and prevalence of fake social media profiles with AI-generated faces.](http://arxiv.org/abs/2401.02627) | 本文分析了使用AI生成的人脸作为头像的伪造社交媒体账户，发现它们用于传播诈骗、垃圾信息以及放大协同信息等不真实活动。通过开发一种有效的方法，作者估计使用GAN生成的面孔的账户普遍性下限在0.021%到0.044%之间，约为每日活跃账户10K个。 |
| [^6] | [FUTURE-AI: International consensus guideline for trustworthy and deployable artificial intelligence in healthcare.](http://arxiv.org/abs/2309.12325) | FUTURE-AI是第一个国际共识框架，为医疗保健领域的可信AI工具开发和部署提供指导原则和最佳实践。 |

# 详细

[^1]: 从表现性伤害到服务质量伤害:羊驼2安全保障的案例研究

    From Representational Harms to Quality-of-Service Harms: A Case Study on Llama 2 Safety Safeguards

    [https://arxiv.org/abs/2403.13213](https://arxiv.org/abs/2403.13213)

    本文探讨了针对表现性伤害和服务质量伤害的羊驼2安全保障措施的有效性，并指出了大型语言模型在实用性和安全性之间的权衡关系。

    

    近期大型语言模型（LLM）的进展导致它们在各个领域被广泛采用。然而，这些进步也引入了额外的安全风险，并引发了对其对已经边缘化人群的不利影响的担忧。尽管存在越来越多的减轻措施来开发安全保障措施，比如监督式的安全定向微调和利用来自人类反馈的安全强化学习，但关于这些模型的安全性和内在偏见仍存在多重关注。此外，先前的研究已经证明，为了安全而优化的模型通常会展示夸大的安全行为，比如出于预防措施而倾向于不回应某些请求。因此，文献中已经记录了这些模型在实用性和安全性之间的明显权衡。在本文中，我们进一步研究了安全措施的有效性，通过评估...

    arXiv:2403.13213v1 Announce Type: cross  Abstract: Recent progress in large language models (LLMs) has led to their widespread adoption in various domains. However, these advancements have also introduced additional safety risks and raised concerns regarding their detrimental impact on already marginalized populations. Despite growing mitigation efforts to develop safety safeguards, such as supervised safety-oriented fine-tuning and leveraging safe reinforcement learning from human feedback, multiple concerns regarding the safety and ingrained biases in these models remain. Furthermore, previous work has demonstrated that models optimized for safety often display exaggerated safety behaviors, such as a tendency to refrain from responding to certain requests as a precautionary measure. As such, a clear trade-off between the helpfulness and safety of these models has been documented in the literature. In this paper, we further investigate the effectiveness of safety measures by evaluatin
    
[^2]: 一个问题中心的多专家对比学习框架，用于提高深度序列知识追踪模型的准确性和可解释性

    A Question-centric Multi-experts Contrastive Learning Framework for Improving the Accuracy and Interpretability of Deep Sequential Knowledge Tracing Models

    [https://arxiv.org/abs/2403.07322](https://arxiv.org/abs/2403.07322)

    通过一个问题中心的多专家对比学习框架，提高深度序列知识追踪模型的准确性和可解释性，解决了知识追踪中个体问题信息建模和模型预测结果解释的重要挑战

    

    知识追踪在通过分析学生历史学习过程来预测其未来表现中发挥着至关重要的作用。深度神经网络在解决知识追踪问题方面展现出巨大潜力。然而，将深度学习技术应用于模拟知识追踪过程仍然存在一些重要挑战。第一个挑战在于将问题的个体信息融入建模中。这很关键，因为尽管问题共享相同的知识组件（KC），但学生对同质问题的知识习得可以有显著差异。第二个挑战在于解释现有基于深度学习的知识追踪模型的预测结果。在真实应用中，虽然可能并不需要完全透明和可解释的模型参数，但关键是以老师能理解的方式呈现模型的预测结果。

    arXiv:2403.07322v1 Announce Type: cross  Abstract: Knowledge tracing (KT) plays a crucial role in predicting students' future performance by analyzing their historical learning processes. Deep neural networks (DNNs) have shown great potential in solving the KT problem. However, there still exist some important challenges when applying deep learning techniques to model the KT process. The first challenge lies in taking the individual information of the question into modeling. This is crucial because, despite questions sharing the same knowledge component (KC), students' knowledge acquisition on homogeneous questions can vary significantly. The second challenge lies in interpreting the prediction results from existing deep learning-based KT models. In real-world applications, while it may not be necessary to have complete transparency and interpretability of the model parameters, it is crucial to present the model's prediction results in a manner that teachers find interpretable. This ma
    
[^3]: 通过监督预训练和重要性机制微调改进低资源知识追踪任务

    Improving Low-Resource Knowledge Tracing Tasks by Supervised Pre-training and Importance Mechanism Fine-tuning

    [https://arxiv.org/abs/2403.06725](https://arxiv.org/abs/2403.06725)

    本文提出了名为LoReKT的低资源知识追踪框架，通过监督预训练和微调重要性机制，旨在从丰富资源的KT数据集中学习可转移的参数和表示来改进低资源知识追踪任务。

    

    知识追踪（KT）旨在基于学生的历史互动来估计他们的知识掌握程度。最近，基于深度学习的KT（DLKT）方法在KT任务中取得了令人印象深刻的表现。然而，由于各种原因，如预算限制和隐私问题，许多实际场景中观察到的互动非常有限，即低资源KT数据集。直接在低资源KT数据集上训练DLKT模型可能会导致过拟合，并且很难选择适当的深度神经架构。因此，在本文中，我们提出了一个名为LoReKT的低资源KT框架来应对上述挑战。受盛行的“预训练和微调”范式的启发，我们旨在在预训练阶段从丰富资源的KT数据集中学习可转移的参数和表示。

    arXiv:2403.06725v1 Announce Type: cross  Abstract: Knowledge tracing (KT) aims to estimate student's knowledge mastery based on their historical interactions. Recently, the deep learning based KT (DLKT) approaches have achieved impressive performance in the KT task. These DLKT models heavily rely on the large number of available student interactions. However, due to various reasons such as budget constraints and privacy concerns, observed interactions are very limited in many real-world scenarios, a.k.a, low-resource KT datasets. Directly training a DLKT model on a low-resource KT dataset may lead to overfitting and it is difficult to choose the appropriate deep neural architecture. Therefore, in this paper, we propose a low-resource KT framework called LoReKT to address above challenges. Inspired by the prevalent "pre-training and fine-tuning" paradigm, we aim to learn transferable parameters and representations from rich-resource KT datasets during the pre-training stage and subseque
    
[^4]: AI思想如何影响人类思想的创造力、多样性和进化：来自一个大规模动态实验的证据

    How AI Ideas Affect the Creativity, Diversity, and Evolution of Human Ideas: Evidence From a Large, Dynamic Experiment. (arXiv:2401.13481v1 [cs.CY])

    [http://arxiv.org/abs/2401.13481](http://arxiv.org/abs/2401.13481)

    AI思想对个体创造力没有影响，但增加了整体思想多样性的数量和变化速率。

    

    大规模语言模型输出的接触正在迅速增加。观看到AI生成的思想将如何影响人类思想？我们进行了一个实验（800+参与者，40+个国家），参与者观看了来自ChatGPT或之前实验参与者的创意思想，然后进行了自己的创意思考。我们变化了AI生成示例的数量（无、低、高曝光）以及示例是否标记为“AI”（披露）。我们的动态实验设计 - 在同一实验条件下，使用之前参与者的思想作为未来参与者的刺激 - 模拟了文化创造的相互依赖过程：创造性思想建立在之前的思想基础上。因此，我们捕捉到了LLM“在文化循环中”的复合效应。我们发现高AI曝光（但不是低AI曝光）并没有影响个人思想的创造力，但增加了整体思想多样性的平均数量和变化速率。AI使思想多样性的累积效应增强了。

    Exposure to large language model output is rapidly increasing. How will seeing AI-generated ideas affect human ideas? We conducted an experiment (800+ participants, 40+ countries) where participants viewed creative ideas that were from ChatGPT or prior experimental participants and then brainstormed their own idea. We varied the number of AI-generated examples (none, low, or high exposure) and if the examples were labeled as 'AI' (disclosure). Our dynamic experiment design -- ideas from prior participants in an experimental condition are used as stimuli for future participants in the same experimental condition -- mimics the interdependent process of cultural creation: creative ideas are built upon prior ideas. Hence, we capture the compounding effects of having LLMs 'in the culture loop'. We find that high AI exposure (but not low AI exposure) did not affect the creativity of individual ideas but did increase the average amount and rate of change of collective idea diversity. AI made 
    
[^5]: 伪造社交媒体账户的特点和普遍性，使用的是AI生成的面孔

    Characteristics and prevalence of fake social media profiles with AI-generated faces. (arXiv:2401.02627v1 [cs.CY])

    [http://arxiv.org/abs/2401.02627](http://arxiv.org/abs/2401.02627)

    本文分析了使用AI生成的人脸作为头像的伪造社交媒体账户，发现它们用于传播诈骗、垃圾信息以及放大协同信息等不真实活动。通过开发一种有效的方法，作者估计使用GAN生成的面孔的账户普遍性下限在0.021%到0.044%之间，约为每日活跃账户10K个。

    

    最近人工智能生成模型的进步引发了对其可能创建出逼真伪造社交媒体账户的担忧，但缺乏实证证据。本文对使用生成对抗网络（GANs）生成的人脸作为头像的Twitter(X)账户进行了系统分析。我们提供了一个包含1,353个此类账户的数据集，并展示了它们用于传播诈骗、垃圾信息以及放大协同信息等不真实活动。通过利用GAN生成的面孔的一个特征——眼睛的一致位置，并与人工注释相结合，我们设计了一种有效的方法来识别野外中使用GAN生成的账户。将该方法应用于随机样本的活跃Twitter用户中，我们估计使用GAN生成的面孔的账户普遍性下限在0.021%到0.044%之间，约为每日活跃账户10K个。这些发现突显了多模式账户对于新兴威胁的重要性。

    Recent advancements in generative artificial intelligence (AI) have raised concerns about their potential to create convincing fake social media accounts, but empirical evidence is lacking. In this paper, we present a systematic analysis of Twitter(X) accounts using human faces generated by Generative Adversarial Networks (GANs) for their profile pictures. We present a dataset of 1,353 such accounts and show that they are used to spread scams, spam, and amplify coordinated messages, among other inauthentic activities. Leveraging a feature of GAN-generated faces -- consistent eye placement -- and supplementing it with human annotation, we devise an effective method for identifying GAN-generated profiles in the wild. Applying this method to a random sample of active Twitter users, we estimate a lower bound for the prevalence of profiles using GAN-generated faces between 0.021% and 0.044% -- around 10K daily active accounts. These findings underscore the emerging threats posed by multimod
    
[^6]: FUTURE-AI：在医疗保健领域的可信和可部署人工智能的国际共识指南

    FUTURE-AI: International consensus guideline for trustworthy and deployable artificial intelligence in healthcare. (arXiv:2309.12325v1 [cs.CY])

    [http://arxiv.org/abs/2309.12325](http://arxiv.org/abs/2309.12325)

    FUTURE-AI是第一个国际共识框架，为医疗保健领域的可信AI工具开发和部署提供指导原则和最佳实践。

    

    尽管在医学和医疗保健领域人工智能（AI）取得了重大进展，但AI技术在现实临床实践中的部署和采用仍受限。近年来，人们对医疗AI的技术、临床、伦理和法律风险提出了关注。为了增加在现实世界中的采用，医疗AI工具必须得到患者、临床医生、健康组织和当局的信任和接受。本文描述了FUTURE-AI指南作为第一个用于指导医疗保健领域可信AI工具开发和部署的国际共识框架。FUTURE-AI联盟成立于2021年，目前包括来自51个国家的118位跨学科专家，代表了所有大洲，包括AI科学家、临床医生、伦理学家和社会科学家。在为期两年的时间里，联盟通过迭代过程定义了可信AI的指导原则和最佳实践，其中包括

    Despite major advances in artificial intelligence (AI) for medicine and healthcare, the deployment and adoption of AI technologies remain limited in real-world clinical practice. In recent years, concerns have been raised about the technical, clinical, ethical and legal risks associated with medical AI. To increase real world adoption, it is essential that medical AI tools are trusted and accepted by patients, clinicians, health organisations and authorities. This work describes the FUTURE-AI guideline as the first international consensus framework for guiding the development and deployment of trustworthy AI tools in healthcare. The FUTURE-AI consortium was founded in 2021 and currently comprises 118 inter-disciplinary experts from 51 countries representing all continents, including AI scientists, clinicians, ethicists, and social scientists. Over a two-year period, the consortium defined guiding principles and best practices for trustworthy AI through an iterative process comprising a
    

