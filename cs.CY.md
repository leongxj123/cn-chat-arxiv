# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Design2Code: How Far Are We From Automating Front-End Engineering?](https://arxiv.org/abs/2403.03163) | 生成式人工智能在多模态理解和代码生成方面取得了突破，提出了Design2Code任务并进行了全面基准测试，展示了多模态LLMs直接将视觉设计转换为代码实现的能力。 |

# 详细

[^1]: Design2Code：我们离自动化前端工程有多远？

    Design2Code: How Far Are We From Automating Front-End Engineering?

    [https://arxiv.org/abs/2403.03163](https://arxiv.org/abs/2403.03163)

    生成式人工智能在多模态理解和代码生成方面取得了突破，提出了Design2Code任务并进行了全面基准测试，展示了多模态LLMs直接将视觉设计转换为代码实现的能力。

    

    近年来，生成式人工智能在多模态理解和代码生成方面取得了突飞猛进的进展，实现了前所未有的能力。这可以实现一种新的前端开发范式，其中多模态LLMs可能直接将视觉设计转换为代码实现。本文将这一过程形式化为Design2Code任务，并进行全面基准测试。我们手动策划了一个包含484个多样化真实网页的基准测试用例，并开发了一套自动评估指标，以评估当前多模态LLMs能否生成直接渲染为给定参考网页的代码实现，以输入为屏幕截图。我们还结合了全面的人工评估。我们开发了一套多模态提示方法，并展示了它们在GPT-4V和Gemini Pro Vision上的有效性。我们进一步对一个开源的Design2Code-18B模型进行了微调。

    arXiv:2403.03163v1 Announce Type: new  Abstract: Generative AI has made rapid advancements in recent years, achieving unprecedented capabilities in multimodal understanding and code generation. This can enable a new paradigm of front-end development, in which multimodal LLMs might directly convert visual designs into code implementations. In this work, we formalize this as a Design2Code task and conduct comprehensive benchmarking. Specifically, we manually curate a benchmark of 484 diverse real-world webpages as test cases and develop a set of automatic evaluation metrics to assess how well current multimodal LLMs can generate the code implementations that directly render into the given reference webpages, given the screenshots as input. We also complement automatic metrics with comprehensive human evaluations. We develop a suite of multimodal prompting methods and show their effectiveness on GPT-4V and Gemini Pro Vision. We further finetune an open-source Design2Code-18B model that su
    

