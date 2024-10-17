# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MDB: Interactively Querying Datasets and Models.](http://arxiv.org/abs/2308.06686) | MDB是一个调试框架，用于互动查询数据集和模型。它通过集成函数式编程与关系代数，能够快速迭代和优化查询，发现和描述错误和模型行为。实验证明，MDB比其他工具能够实现更快的查询速度加快和查询长度缩短。 |

# 详细

[^1]: MDB：互动查询数据集和模型

    MDB: Interactively Querying Datasets and Models. (arXiv:2308.06686v1 [cs.DB])

    [http://arxiv.org/abs/2308.06686](http://arxiv.org/abs/2308.06686)

    MDB是一个调试框架，用于互动查询数据集和模型。它通过集成函数式编程与关系代数，能够快速迭代和优化查询，发现和描述错误和模型行为。实验证明，MDB比其他工具能够实现更快的查询速度加快和查询长度缩短。

    

    随着模型的训练和部署，开发者需要能够系统地调试在机器学习流程中出现的错误。我们提出了MDB，一个用于互动查询数据集和模型的调试框架。MDB通过将函数式编程与关系代数结合起来，构建了一个对数据集和模型预测的数据库进行表达性查询的工具。查询可重用且易于修改，使得调试人员能够快速迭代和优化查询，以发现和描述错误和模型行为。我们在目标检测、偏差发现、图像分类和数据填充任务中评估了MDB在自动驾驶视频、大型语言模型和医疗记录上的性能。我们的实验证明，MDB比其他基准测试工具能够实现最高10倍的查询速度加快和40%的查询长度缩短。在用户研究中，我们发现开发者能够成功构建复杂查询来描述机器学习模型的错误。

    As models are trained and deployed, developers need to be able to systematically debug errors that emerge in the machine learning pipeline. We present MDB, a debugging framework for interactively querying datasets and models. MDB integrates functional programming with relational algebra to build expressive queries over a database of datasets and model predictions. Queries are reusable and easily modified, enabling debuggers to rapidly iterate and refine queries to discover and characterize errors and model behaviors. We evaluate MDB on object detection, bias discovery, image classification, and data imputation tasks across self-driving videos, large language models, and medical records. Our experiments show that MDB enables up to 10x faster and 40\% shorter queries than other baselines. In a user study, we find developers can successfully construct complex queries that describe errors of machine learning models.
    

