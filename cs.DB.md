# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SMUTF: Schema Matching Using Generative Tags and Hybrid Features](https://arxiv.org/abs/2402.01685) | SMUTF是一种用于大规模表格数据模式匹配的独特方法，通过结合基于规则的特征工程、预训练语言模型和生成式大语言模型，并使用生成标签提高匹配效果。同时，作者开发并开源了HDXSM数据集来解决现有数据集不足的问题。 |

# 详细

[^1]: SMUTF：使用生成标签和混合特征的模式匹配方法

    SMUTF: Schema Matching Using Generative Tags and Hybrid Features

    [https://arxiv.org/abs/2402.01685](https://arxiv.org/abs/2402.01685)

    SMUTF是一种用于大规模表格数据模式匹配的独特方法，通过结合基于规则的特征工程、预训练语言模型和生成式大语言模型，并使用生成标签提高匹配效果。同时，作者开发并开源了HDXSM数据集来解决现有数据集不足的问题。

    

    我们引入了SMUTF，一种用于大规模表格数据模式匹配的独特方法，该方法假设在开放域任务中，监督学习不会影响性能，从而实现了有效的跨域匹配。这个系统独特地结合了基于规则的特征工程、预训练语言模型和生成式大语言模型。受人道主义交换语言的启发，我们使用“生成标签”为每个数据列部署了创新的适应性，提高了模式匹配的效果。SMUTF具有广泛的灵活性，可以与任何现有的预训练嵌入、分类方法和生成模型无缝配合使用。鉴于模式匹配缺乏广泛的公开数据集，我们已经创建并开源了HDXSM数据集，该数据集来自公共人道主义数据，我们相信这是目前最全面的模式匹配数据集。

    We introduce SMUTF, a unique approach for large-scale tabular data schema matching (SM), which assumes that supervised learning does not affect performance in open-domain tasks, thereby enabling effective cross-domain matching. This system uniquely combines rule-based feature engineering, pre-trained language models, and generative large language models. In an innovative adaptation inspired by the Humanitarian Exchange Language, we deploy 'generative tags' for each data column, enhancing the effectiveness of SM. SMUTF exhibits extensive versatility, working seamlessly with any pre-existing pre-trained embeddings, classification methods, and generative models.   Recognizing the lack of extensive, publicly available datasets for SM, we have created and open-sourced the HDXSM dataset from the public humanitarian data. We believe this to be the most exhaustive SM dataset currently available. In evaluations across various public datasets and the novel HDXSM dataset, SMUTF demonstrated excep
    

