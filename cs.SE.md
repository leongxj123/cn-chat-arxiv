# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Exploring Safety Generalization Challenges of Large Language Models via Code](https://arxiv.org/abs/2403.07865) | 本论文引入了CodeAttack框架用于测试大型语言模型的安全泛化，研究发现GPT-4、Claude-2和Llama-2系列等最新模型存在代码输入的安全漏洞。 |
| [^2] | [AIOptimizer -- A reinforcement learning-based software performance optimisation prototype for cost minimisation.](http://arxiv.org/abs/2307.07846) | AIOptimizer是一种基于强化学习的软件性能优化工具原型，旨在实现成本最小化。它使用强化学习驱动的推荐系统来改善软件系统的效率和可负担性，并突出了准确性、适应性、可扩展性和用户友好性等设计因素。AIOptimizer还提供故障识别、成本优化建议、效率预测和协作等功能，并使用基于强化学习的推荐引擎进行成本优化。 |

# 详细

[^1]: 通过代码探索大型语言模型的安全泛化挑战

    Exploring Safety Generalization Challenges of Large Language Models via Code

    [https://arxiv.org/abs/2403.07865](https://arxiv.org/abs/2403.07865)

    本论文引入了CodeAttack框架用于测试大型语言模型的安全泛化，研究发现GPT-4、Claude-2和Llama-2系列等最新模型存在代码输入的安全漏洞。

    

    大型语言模型（LLMs）的快速发展带来了自然语言处理方面的显著能力，但也引发了人们对它们潜在误用的担忧。本文引入了CodeAttack，一个将自然语言输入转换为代码输入的框架，为测试LLMs的安全泛化提供了一个新颖的环境。我们对包括GPT-4、Claude-2和Llama-2系列在内的最新LLMs进行了全面研究，发现这些模型对于代码输入存在共同的安全漏洞：CodeAttack在超过80%的时间内始终绕过所有模型的安全保护。

    arXiv:2403.07865v1 Announce Type: cross  Abstract: The rapid advancement of Large Language Models (LLMs) has brought about remarkable capabilities in natural language processing but also raised concerns about their potential misuse. While strategies like supervised fine-tuning and reinforcement learning from human feedback have enhanced their safety, these methods primarily focus on natural languages, which may not generalize to other domains. This paper introduces CodeAttack, a framework that transforms natural language inputs into code inputs, presenting a novel environment for testing the safety generalization of LLMs. Our comprehensive studies on state-of-the-art LLMs including GPT-4, Claude-2, and Llama-2 series reveal a common safety vulnerability of these models against code input: CodeAttack consistently bypasses the safety guardrails of all models more than 80\% of the time. Furthermore, we find that a larger distribution gap between CodeAttack and natural language leads to we
    
[^2]: AIOptimizer ——基于强化学习的软件性能优化原型，旨在实现成本最小化

    AIOptimizer -- A reinforcement learning-based software performance optimisation prototype for cost minimisation. (arXiv:2307.07846v1 [cs.SE])

    [http://arxiv.org/abs/2307.07846](http://arxiv.org/abs/2307.07846)

    AIOptimizer是一种基于强化学习的软件性能优化工具原型，旨在实现成本最小化。它使用强化学习驱动的推荐系统来改善软件系统的效率和可负担性，并突出了准确性、适应性、可扩展性和用户友好性等设计因素。AIOptimizer还提供故障识别、成本优化建议、效率预测和协作等功能，并使用基于强化学习的推荐引擎进行成本优化。

    

    本研究文章介绍了AIOptimizer，一个基于成本降低的软件性能优化工具的原型。AIOptimizer使用强化学习驱动的推荐系统来改善软件系统的效率和可负担性。本文强调了AIOptimizer的设计因素，如准确性、适应性、可扩展性和用户友好性。为了提供有效的用户中心的性能优化解决方案，它强调了模块化设计、数据收集技术、持续学习和弹性集成的使用。本文还调查了AIOptimizer的特性，如故障识别、成本优化建议、效率预测和协作。此外，本文还探讨了几个软件开发生命周期模型，并介绍了AIOptimizer使用基于强化学习的推荐引擎进行成本优化。本研究旨在突出AIOptimizer作为一种利用先进技术进行成本优化的原型。

    This research article introduces AIOptimizer, a prototype for a software performance optimisation tool based on cost reduction. AIOptimizer uses a recommendation system driven by reinforcement learning to improve software system efficiency and affordability. The paper highlights AIOptimizer's design factors, such as accuracy, adaptability, scalability, and user-friendliness. To provide effective and user-centric performance optimisation solutions, it emphasises the use of a modular design, data gathering techniques, continuous learning, and resilient integration. The article also investigates AIOptimizer features such as fault identification, cost optimisation recommendations, efficiency prediction, and cooperation. Furthermore, it explores several software development life cycle models and introduces AIOptimizer uses a reinforcement learning-based recommendation engine for cost optimisation. The purpose of this research study is to highlight AIOptimizer as a prototype that uses advanc
    

