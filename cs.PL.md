# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ChipGPT: How far are we from natural language hardware design.](http://arxiv.org/abs/2305.14019) | 这篇论文介绍了ChipGPT，一个自动化设计环境，它利用大型语言模型从自然语言规范生成硬件逻辑设计，并展示了与人工设计性能相媲美的结果，且可节省超过75％的编码时间。 |

# 详细

[^1]: ChipGPT: 远离自然语言硬件设计还有多远

    ChipGPT: How far are we from natural language hardware design. (arXiv:2305.14019v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2305.14019](http://arxiv.org/abs/2305.14019)

    这篇论文介绍了ChipGPT，一个自动化设计环境，它利用大型语言模型从自然语言规范生成硬件逻辑设计，并展示了与人工设计性能相媲美的结果，且可节省超过75％的编码时间。

    

    随着大型语言模型（LLMs）如ChatGPT展示了前所未有的机器智能，它在通过自然语言交互来协助硬件工程师实现更高效的逻辑设计方面也表现出极佳的性能。为了评估LLMs协助硬件设计过程的潜力，本文尝试演示一个自动化设计环境，该环境利用LLMs从自然语言规范生成硬件逻辑设计。为了实现更易用且更高效的芯片开发流程，我们提出了一种基于LLMs的可扩展的四阶段零代码逻辑设计框架，无需重新训练或微调。首先，演示版本ChipGPT通过为LLM生成提示开始，然后产生初始Verilog程序。 其次，输出管理器纠正和优化这些程序，然后将它们收集到最终的设计空间中。最后，ChipGPT将在此空间中搜索以选择符合目标指标的最优设计。评估表明，由ChipGPT设计的逻辑电路的性能与人工设计的性能相当，并且整个过程节省了超过75％的编码时间。

    As large language models (LLMs) like ChatGPT exhibited unprecedented machine intelligence, it also shows great performance in assisting hardware engineers to realize higher-efficiency logic design via natural language interaction. To estimate the potential of the hardware design process assisted by LLMs, this work attempts to demonstrate an automated design environment that explores LLMs to generate hardware logic designs from natural language specifications. To realize a more accessible and efficient chip development flow, we present a scalable four-stage zero-code logic design framework based on LLMs without retraining or finetuning. At first, the demo, ChipGPT, begins by generating prompts for the LLM, which then produces initial Verilog programs. Second, an output manager corrects and optimizes these programs before collecting them into the final design space. Eventually, ChipGPT will search through this space to select the optimal design under the target metrics. The evaluation sh
    

