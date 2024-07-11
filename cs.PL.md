# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Data is all you need: Finetuning LLMs for Chip Design via an Automated design-data augmentation framework](https://arxiv.org/abs/2403.11202) | 提出了一种自动设计数据增强框架，以优化大型语言模型在芯片设计中的应用能力，并解决了Verilog数据匮乏和训练数据准备时间长的问题 |

# 详细

[^1]: 数据就是你需要的一切：通过自动设计数据增强框架对LLM进行芯片设计微调

    Data is all you need: Finetuning LLMs for Chip Design via an Automated design-data augmentation framework

    [https://arxiv.org/abs/2403.11202](https://arxiv.org/abs/2403.11202)

    提出了一种自动设计数据增强框架，以优化大型语言模型在芯片设计中的应用能力，并解决了Verilog数据匮乏和训练数据准备时间长的问题

    

    最近大型语言模型的进展表明，它们在高层提示下自动生成硬件描述语言（HDL）代码的潜力。研究人员利用微调来增强这些大型语言模型（LLMs）在芯片设计领域的能力。然而，缺乏Verilog数据阻碍了LLMs在Verilog生成质量上的进一步提升。此外，缺少Verilog和电子设计自动化（EDA）脚本数据增强框架显着增加了为LLM训练器准备训练数据集所需的时间。本文提出了一种自动设计数据增强框架，它生成与Verilog和EDA脚本对齐的大量高质量自然语言。对于Verilog生成，它将Verilog文件转换为抽象语法树，然后将节点映射到具有预定义模板的自然语言。对于Verilog修复，它

    arXiv:2403.11202v1 Announce Type: cross  Abstract: Recent advances in large language models have demonstrated their potential for automated generation of hardware description language (HDL) code from high-level prompts. Researchers have utilized fine-tuning to enhance the ability of these large language models (LLMs) in the field of Chip Design. However, the lack of Verilog data hinders further improvement in the quality of Verilog generation by LLMs. Additionally, the absence of a Verilog and Electronic Design Automation (EDA) script data augmentation framework significantly increases the time required to prepare the training dataset for LLM trainers. This paper proposes an automated design-data augmentation framework, which generates high-volume and high-quality natural language aligned with Verilog and EDA scripts. For Verilog generation, it translates Verilog files to an abstract syntax tree and then maps nodes to natural language with a predefined template. For Verilog repair, it 
    

