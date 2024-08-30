# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Understanding how Differentially Private Generative Models Spend their Privacy Budget.](http://arxiv.org/abs/2305.10994) | 本文分析了采用差分隐私训练的生成模型如何分配隐私预算，以及影响分配的因素。使用不同的模型适合于不同的任务和设置。 |

# 详细

[^1]: 理解差分隐私生成模型如何使用隐私预算

    Understanding how Differentially Private Generative Models Spend their Privacy Budget. (arXiv:2305.10994v1 [cs.LG])

    [http://arxiv.org/abs/2305.10994](http://arxiv.org/abs/2305.10994)

    本文分析了采用差分隐私训练的生成模型如何分配隐私预算，以及影响分配的因素。使用不同的模型适合于不同的任务和设置。

    

    采用差分隐私训练的生成模型被广泛应用于产生合成数据，同时减少隐私风险。但是在不同的应用场景中找到最适合的模型，需要权衡它们之间的隐私-效用关系。本文针对表格数据，分析了DP生成模型如何分配隐私预算，并探讨了影响隐私预算分配的主要因素。我们对图形和深度生成模型进行了广泛的评估，揭示了不同模型适用于不同设置和任务的独特特征。

    Generative models trained with Differential Privacy (DP) are increasingly used to produce synthetic data while reducing privacy risks. Navigating their specific privacy-utility tradeoffs makes it challenging to determine which models would work best for specific settings/tasks. In this paper, we fill this gap in the context of tabular data by analyzing how DP generative models distribute privacy budgets across rows and columns, arguably the main source of utility degradation. We examine the main factors contributing to how privacy budgets are spent, including underlying modeling techniques, DP mechanisms, and data dimensionality.  Our extensive evaluation of both graphical and deep generative models sheds light on the distinctive features that render them suitable for different settings and tasks. We show that graphical models distribute the privacy budget horizontally and thus cannot handle relatively wide datasets while the performance on the task they were optimized for monotonicall
    

