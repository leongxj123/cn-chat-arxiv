# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Pre-Training Representations of Binary Code Using Contrastive Learning.](http://arxiv.org/abs/2210.05102) | 提出了一种使用对比学习预训练二进制代码表示的方法，可以将源代码和注释信息纳入二进制代码的表示学习中，对于反向工程和计算机安全任务有重要意义。 |

# 详细

[^1]: 使用对比学习预训练二进制代码表示

    Pre-Training Representations of Binary Code Using Contrastive Learning. (arXiv:2210.05102v2 [cs.SE] UPDATED)

    [http://arxiv.org/abs/2210.05102](http://arxiv.org/abs/2210.05102)

    提出了一种使用对比学习预训练二进制代码表示的方法，可以将源代码和注释信息纳入二进制代码的表示学习中，对于反向工程和计算机安全任务有重要意义。

    

    编译后的软件以可执行的二进制代码形式交付。开发人员编写源代码来表达软件的语义，但编译器将其转换为CPU可以直接执行的二进制格式。因此，二进制代码分析对于反向工程和计算机安全任务等没有源代码的应用程序至关重要。然而，与包含丰富语义信息的源代码和自然语言不同，二进制代码通常难以理解和分析。虽然现有的工作使用AI模型辅助源代码分析，但很少有研究考虑二进制代码。在本文中，我们提出了一种将源代码和注释信息纳入二进制代码进行表示学习的对比学习模型，称为COMBO。具体而言，我们在COMBO中提出了三个组件：（1）用于冷启动预训练的主要对比学习方法，（2）用于将源代码和注释信息插入到二进制代码中的单纯插值方法。

    Compiled software is delivered as executable binary code. Developers write source code to express the software semantics, but the compiler converts it to a binary format that the CPU can directly execute. Therefore, binary code analysis is critical to applications in reverse engineering and computer security tasks where source code is not available. However, unlike source code and natural language that contain rich semantic information, binary code is typically difficult for human engineers to understand and analyze. While existing work uses AI models to assist source code analysis, few studies have considered binary code. In this paper, we propose a COntrastive learning Model for Binary cOde Analysis, or COMBO, that incorporates source code and comment information into binary code during representation learning. Specifically, we present three components in COMBO: (1) a primary contrastive learning method for cold-start pre-training, (2) a simplex interpolation method to incorporate so
    

