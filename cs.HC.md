# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards the new XAI: A Hypothesis-Driven Approach to Decision Support Using Evidence](https://rss.arxiv.org/abs/2402.01292) | 本文介绍并评估了一种基于证据权重框架的假设驱动可解释人工智能方法，通过提供支持或驳斥假设的证据来增加决策准确性和减少依赖程度。 |
| [^2] | [Non-discrimination Criteria for Generative Language Models](https://arxiv.org/abs/2403.08564) | 本文研究如何在生成式语言模型中识别和量化性别偏见，提出了三个生成式人工智能的非歧视标准并设计了相应的提示。 |
| [^3] | [Optimizing Delegation in Collaborative Human-AI Hybrid Teams](https://arxiv.org/abs/2402.05605) | 本论文提出了一种优化协作的人工智能-人类混合团队授权的框架，通过引入AI经理（通过强化学习）作为团队的外部观察者，学习团队代理人的行为模型并选择最佳的控制代理人。 |

# 详细

[^1]: 迈向新的可解释人工智能：通过证据支持的假设驱动方法的决策支持

    Towards the new XAI: A Hypothesis-Driven Approach to Decision Support Using Evidence

    [https://rss.arxiv.org/abs/2402.01292](https://rss.arxiv.org/abs/2402.01292)

    本文介绍并评估了一种基于证据权重框架的假设驱动可解释人工智能方法，通过提供支持或驳斥假设的证据来增加决策准确性和减少依赖程度。

    

    之前关于AI辅助人类决策的研究探索了几种不同的可解释人工智能（XAI）方法。最近的一篇论文提出了一种范式转变，呼吁通过一个称为评价型AI的概念框架来进行假设驱动的XAI，该框架为人们提供支持或驳斥假设的证据，而不一定给出决策辅助推荐。在本文中，我们描述并评估了一种基于证据权重（WoE）框架的假设驱动XAI方法，该方法为给定的假设生成正面和负面证据。通过人类行为实验，我们展示了我们的假设驱动方法提高了决策准确性，与推荐驱动方法和仅AI解释基线相比减少了依赖程度，但相对于推荐驱动方法，在依赖程度下降方面略微增加。此外，我们还展示了参与者在使用我们的假设驱动方法时与两个基线的方式存在实质性的差异。

    Prior research on AI-assisted human decision-making has explored several different explainable AI (XAI) approaches. A recent paper has proposed a paradigm shift calling for hypothesis-driven XAI through a conceptual framework called evaluative AI that gives people evidence that supports or refutes hypotheses without necessarily giving a decision-aid recommendation. In this paper we describe and evaluate an approach for hypothesis-driven XAI based on the Weight of Evidence (WoE) framework, which generates both positive and negative evidence for a given hypothesis. Through human behavioural experiments, we show that our hypothesis-driven approach increases decision accuracy, reduces reliance compared to a recommendation-driven approach and an AI-explanation-only baseline, but with a small increase in under-reliance compared to the recommendation-driven approach. Further, we show that participants used our hypothesis-driven approach in a materially different way to the two baselines.
    
[^2]: 生成语言模型的非歧视标准

    Non-discrimination Criteria for Generative Language Models

    [https://arxiv.org/abs/2403.08564](https://arxiv.org/abs/2403.08564)

    本文研究如何在生成式语言模型中识别和量化性别偏见，提出了三个生成式人工智能的非歧视标准并设计了相应的提示。

    

    近年来，生成式人工智能，如大型语言模型，经历了快速发展。随着这些模型越来越普遍地提供给公众使用，人们开始担心在应用中延续和放大有害偏见的问题。性别刻板印象可能对其针对的个人造成伤害和限制，无论是由误传还是歧视所构成。识别性别偏见作为一种普遍的社会构造，本文研究如何发现和量化生成式语言模型中性别偏见的存在。具体而言，我们推导出三个来自分类的著名非歧视标准的生成式人工智能类比，即独立性、分离性和充分性。为了展示这些标准的作用，我们设计了针对每个标准的提示，重点关注职业性别刻板印象，具体利用医学测试来在生成式人工智能背景中引入基本事实。

    arXiv:2403.08564v1 Announce Type: cross  Abstract: Within recent years, generative AI, such as large language models, has undergone rapid development. As these models become increasingly available to the public, concerns arise about perpetuating and amplifying harmful biases in applications. Gender stereotypes can be harmful and limiting for the individuals they target, whether they consist of misrepresentation or discrimination. Recognizing gender bias as a pervasive societal construct, this paper studies how to uncover and quantify the presence of gender biases in generative language models. In particular, we derive generative AI analogues of three well-known non-discrimination criteria from classification, namely independence, separation and sufficiency. To demonstrate these criteria in action, we design prompts for each of the criteria with a focus on occupational gender stereotype, specifically utilizing the medical test to introduce the ground truth in the generative AI context. 
    
[^3]: 优化协作的人工智能-人类混合团队中的授权

    Optimizing Delegation in Collaborative Human-AI Hybrid Teams

    [https://arxiv.org/abs/2402.05605](https://arxiv.org/abs/2402.05605)

    本论文提出了一种优化协作的人工智能-人类混合团队授权的框架，通过引入AI经理（通过强化学习）作为团队的外部观察者，学习团队代理人的行为模型并选择最佳的控制代理人。

    

    当人类和自主系统作为混合团队共同运作时，我们希望确保团队的成功和效率。我们将团队成员称为代理人。在我们提出的框架中，我们解决了混合团队的情况，即在任何时候，只有一个团队成员（控制代理人）被授权为团队的控制者。为了确定最佳的控制代理人选择，我们提出了引入AI经理（通过强化学习）的想法，该经理作为团队的外部观察者学习。经理通过观察代理人的表现和团队所处的环境/世界来学习行为模型，并基于这些观察结果选择出最理想的控制代理人。为了限定经理的任务，我们引入了一组约束条件。经理的约束条件指示团队的可接受运作方式，因此如果团队进入不可接受并需要经理介入的状态，就会违反约束条件。

    When humans and autonomous systems operate together as what we refer to as a hybrid team, we of course wish to ensure the team operates successfully and effectively. We refer to team members as agents. In our proposed framework, we address the case of hybrid teams in which, at any time, only one team member (the control agent) is authorized to act as control for the team. To determine the best selection of a control agent, we propose the addition of an AI manager (via Reinforcement Learning) which learns as an outside observer of the team. The manager learns a model of behavior linking observations of agent performance and the environment/world the team is operating in, and from these observations makes the most desirable selection of a control agent. We restrict the manager task by introducing a set of constraints. The manager constraints indicate acceptable team operation, so a violation occurs if the team enters a condition which is unacceptable and requires manager intervention. To
    

