# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generation of Asset Administration Shell with Large Language Model Agents: Interoperability in Digital Twins with Semantic Node](https://arxiv.org/abs/2403.17209) | 通过大型语言模型代理生成AAS实例模型，实现了在数字孪生中的互操作性，降低了手动创建成本和时间。 |
| [^2] | [A Survey of Neural Code Intelligence: Paradigms, Advances and Beyond](https://arxiv.org/abs/2403.14734) | 神经代码智能领域的调查系统回顾了50多种代表性模型和超过680项相关作品，突出了不同研究阶段的范式和技术转变。 |
| [^3] | [Evaluation of LLMs on Syntax-Aware Code Fill-in-the-Middle Tasks](https://arxiv.org/abs/2403.04814) | 该研究引入了一个新的基准SAFIM用于评估LLMs在代码填空任务上的句法感知完成表现，发现FIM预训练不仅提高了FIM的熟练度，还改善了LLMs的左到右推理，挑战了传统观念并表明预训练方法和数据品质对模型性能的影响更甚于模型大小。 |
| [^4] | [OMPGPT: A Generative Pre-trained Transformer Model for OpenMP.](http://arxiv.org/abs/2401.16445) | OMPGPT是一种为了OpenMP pragma生成而设计的生成式预训练Transformer模型，采用了来自NLP领域的提示工程技术，并创建了一种创新的策略chain-of-OMP。 |
| [^5] | [AST-T5: Structure-Aware Pretraining for Code Generation and Understanding.](http://arxiv.org/abs/2401.03003) | AST-T5是一种结构感知的预训练模型，通过利用抽象语法树（AST）来增强代码生成、转换和理解的能力。它优于其他同等大小的语言模型，并在代码到代码任务中表现出色。 |

# 详细

[^1]: 利用大型语言模型代理生成资产管理外壳：数字孪生和语义节点中的互操作性

    Generation of Asset Administration Shell with Large Language Model Agents: Interoperability in Digital Twins with Semantic Node

    [https://arxiv.org/abs/2403.17209](https://arxiv.org/abs/2403.17209)

    通过大型语言模型代理生成AAS实例模型，实现了在数字孪生中的互操作性，降低了手动创建成本和时间。

    

    这项研究介绍了一种新颖的方法，用于协助在工业4.0背景下为数字孪生建模创建资产管理外壳（AAS）实例，旨在增强智能制造中的互操作性，减少手动工作。我们构建了一个“语义节点”数据结构来捕捉文本数据的语义要义。然后，设计并实现了一个由大型语言模型驱动的系统，用于处理“语义节点”并从文本技术数据生成AAS实例模型。我们的评估表明，有效生成率为62-79%，表明相当比例的手动创建工作可以转换为更容易的验证工作，从而减少创建AAS实例模型的时间和成本。在我们的评估中，对不同LLM的比较分析以及检索增强生成（RAG）机制的深入消融研究提供了有关LLM有效性的见解。

    arXiv:2403.17209v1 Announce Type: new  Abstract: This research introduces a novel approach for assisting the creation of Asset Administration Shell (AAS) instances for digital twin modeling within the context of Industry 4.0, aiming to enhance interoperability in smart manufacturing and reduce manual effort. We construct a "semantic node" data structure to capture the semantic essence of textual data. Then, a system powered by large language models is designed and implemented to process "semantic node" and generate AAS instance models from textual technical data. Our evaluation demonstrates a 62-79% effective generation rate, indicating a substantial proportion of manual creation effort can be converted into easier validation effort, thereby reducing the time and cost in creating AAS instance models. In our evaluation, a comparative analysis of different LLMs and an in-depth ablation study of Retrieval-Augmented Generation (RAG) mechanisms provide insights into the effectiveness of LLM
    
[^2]: 一项神经代码智能的调查：范式、进展与未来

    A Survey of Neural Code Intelligence: Paradigms, Advances and Beyond

    [https://arxiv.org/abs/2403.14734](https://arxiv.org/abs/2403.14734)

    神经代码智能领域的调查系统回顾了50多种代表性模型和超过680项相关作品，突出了不同研究阶段的范式和技术转变。

    

    arXiv:2403.14734v1 公告类型: 跨领域 摘要: 神经代码智能--利用深度学习理解、生成和优化代码--在整个社会上具有巨大的潜力，可产生深远影响。作为自然语言和编程语言之间的桥梁，这一领域在过去几年引起了两个研究社区研究人员的极大关注。本调查系统地和按时间顺序回顾了代码智能方面的进展，包括50多种代表性模型及其变体、20多种任务类别以及超过680项相关作品。我们遵循历史进展，跟踪不同研究阶段的范式转变（例如，从使用循环神经网络对代码建模到大型语言模型时代）。同时，我们重点介绍了不同阶段涵盖的模型、任务和评估的主要技术转变。对于应用，我们

    arXiv:2403.14734v1 Announce Type: cross  Abstract: Neural Code Intelligence -- leveraging deep learning to understand, generate, and optimize code -- holds immense potential for transformative impacts on the whole society. Bridging the gap between Natural Language and Programming Language, this domain has drawn significant attention from researchers in both research communities over the past few years. This survey presents a systematic and chronological review of the advancements in code intelligence, encompassing over 50 representative models and their variants, more than 20 categories of tasks, and an extensive coverage of over 680 related works. We follow the historical progression to trace the paradigm shifts across different research phases (e.g., from modeling code with recurrent neural networks to the era of Large Language Models). Concurrently, we highlight the major technical transitions in models, tasks, and evaluations spanning through different stages. For applications, we 
    
[^3]: 在句法感知代码填空任务上评估LLMs

    Evaluation of LLMs on Syntax-Aware Code Fill-in-the-Middle Tasks

    [https://arxiv.org/abs/2403.04814](https://arxiv.org/abs/2403.04814)

    该研究引入了一个新的基准SAFIM用于评估LLMs在代码填空任务上的句法感知完成表现，发现FIM预训练不仅提高了FIM的熟练度，还改善了LLMs的左到右推理，挑战了传统观念并表明预训练方法和数据品质对模型性能的影响更甚于模型大小。

    

    我们介绍了一种名为Syntax-Aware Fill-In-the-Middle（SAFIM）的新基准，用于评估大型语言模型（LLMs）在代码填空（FIM）任务上的表现。该基准侧重于程序结构的句法感知完成，如代码块和条件表达式，并包括来自多种编程语言的17,720个示例，来源于2022年4月之后的最新代码提交，以最小化数据污染。 SAFIM提供了一个强大的框架，具有各种提示设计和新颖的句法感知后处理技术，有助于在LLMs之间进行准确和公平的比较。我们对15个LLMs进行了全面评估，结果表明FIM预训练不仅提升了FIM的熟练程度，还改进了LLMs的左到右（L2R）推理。我们的发现挑战了传统观念，并表明预训练方法和数据质量对模型性能的影响大于模型大小。因此，SAFIM为未来构建

    arXiv:2403.04814v1 Announce Type: cross  Abstract: We introduce Syntax-Aware Fill-In-the-Middle (SAFIM), a new benchmark for evaluating Large Language Models (LLMs) on the code Fill-in-the-Middle (FIM) task. This benchmark focuses on syntax-aware completions of program structures such as code blocks and conditional expressions, and includes 17,720 examples from multiple programming languages, sourced from recent code submissions after April 2022 to minimize data contamination. SAFIM provides a robust framework with various prompt designs and novel syntax-aware post-processing techniques, facilitating accurate and fair comparisons across LLMs. Our comprehensive evaluation of 15 LLMs shows that FIM pretraining not only enhances FIM proficiency but also improves Left-to-Right (L2R) inference using LLMs. Our findings challenge conventional beliefs and suggest that pretraining methods and data quality have more impact than model size. SAFIM thus serves as a foundational platform for future 
    
[^4]: OMPGPT: 一种用于OpenMP的生成式预训练Transformer模型

    OMPGPT: A Generative Pre-trained Transformer Model for OpenMP. (arXiv:2401.16445v1 [cs.SE])

    [http://arxiv.org/abs/2401.16445](http://arxiv.org/abs/2401.16445)

    OMPGPT是一种为了OpenMP pragma生成而设计的生成式预训练Transformer模型，采用了来自NLP领域的提示工程技术，并创建了一种创新的策略chain-of-OMP。

    

    大型语言模型（LLMs），如ChatGPT等模型，已经在自然语言处理领域引起了革命。随着这一趋势，基于代码的大型语言模型，如StarCoder、WizardCoder和CodeLlama等，已经涌现出来，在大量的代码数据库上进行了广泛的训练。然而，由于设计固有的原因，这些模型主要关注代码生成、代码完成和注释生成等生成任务，以及对多种编程语言的一般支持。虽然代码LLMs的通用能力对许多程序员来说很有用，但高性能计算（HPC）领域具有更窄的需求集，使得更小、更具领域特定的LM成为一个更明智的选择。本文介绍了OMPGPT，这是一种精心设计的新型模型，旨在充分利用语言模型在OpenMP pragma生成方面的固有优势。此外，我们采用并改进了来自NLP领域的提示工程技术，创建了链式OMP（chain-of-OMP），这是一种创新策略。

    Large language models (LLMs), as epitomized by models like ChatGPT, have revolutionized the field of natural language processing (NLP). Along with this trend, code-based large language models such as StarCoder, WizardCoder, and CodeLlama have emerged, trained extensively on vast repositories of code data. Yet, inherent in their design, these models primarily focus on generative tasks like code generation, code completion, and comment generation, and general support for multiple programming languages. While the generic abilities of code LLMs are useful for many programmers, the area of high-performance computing (HPC) has a narrower set of requirements that make a smaller and more domain-specific LM a smarter choice. This paper introduces OMPGPT, a novel model meticulously designed to harness the inherent strengths of language models for OpenMP pragma generation. Furthermore, we adopt and adapt prompt engineering techniques from the NLP domain to create chain-of-OMP, an innovative strat
    
[^5]: AST-T5：面向代码生成和理解的结构感知预训练模型

    AST-T5: Structure-Aware Pretraining for Code Generation and Understanding. (arXiv:2401.03003v1 [cs.SE])

    [http://arxiv.org/abs/2401.03003](http://arxiv.org/abs/2401.03003)

    AST-T5是一种结构感知的预训练模型，通过利用抽象语法树（AST）来增强代码生成、转换和理解的能力。它优于其他同等大小的语言模型，并在代码到代码任务中表现出色。

    

    大型语言模型在代码相关任务中取得了显著进展，然而许多模型将代码视为简单序列，忽略了其结构化特性。我们引入了AST-T5，一种新颖的预训练范式，利用抽象语法树（AST）增强了代码生成、转换和理解。通过动态规划，我们的AST感知分割保留了代码结构，而AST感知跨度破坏目标使模型能够重建各种代码结构。与其他模型不同，AST-T5避免了复杂的程序分析或架构更改，因此可以与任何编码器-解码器Transformer无缝集成。评估结果显示，AST-T5在各种代码相关任务中始终优于同等大小的语言模型。结构感知使得AST-T5在代码到代码任务中特别强大，在Bugs2Fix任务的精确匹配得分上超过CodeT5 2个点，并在CodeXGLUE中的Java-C#转换任务的精确匹配得分上超过CodeT5 3个点。

    Large language models (LLMs) have made significant advancements in code-related tasks, yet many LLMs treat code as simple sequences, neglecting its structured nature. We introduce AST-T5, a novel pretraining paradigm that leverages the Abstract Syntax Tree (AST) for enhanced code generation, transpilation, and understanding. Using dynamic programming, our AST-Aware Segmentation retains code structure, while our AST-Aware Span Corruption objective equips the model to reconstruct various code structures. Unlike other models, AST-T5 avoids intricate program analyses or architectural changes, so it integrates seamlessly with any encoder-decoder Transformer. Evaluations show that AST-T5 consistently outperforms similar-sized LMs across various code-related tasks. Structure-awareness makes AST-T5 particularly powerful in code-to-code tasks, surpassing CodeT5 by 2 points in exact match score for the Bugs2Fix task and by 3 points in exact match score for Java-C# Transpilation in CodeXGLUE. Our
    

