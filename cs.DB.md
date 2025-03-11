# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Roq: Robust Query Optimization Based on a Risk-aware Learned Cost Model.](http://arxiv.org/abs/2401.15210) | Roq是一个基于风险感知学习方法的综合框架，用于实现鲁棒的查询优化。 |

# 详细

[^1]: Roq：基于风险感知学习成本模型的鲁棒查询优化

    Roq: Robust Query Optimization Based on a Risk-aware Learned Cost Model. (arXiv:2401.15210v1 [cs.DB])

    [http://arxiv.org/abs/2401.15210](http://arxiv.org/abs/2401.15210)

    Roq是一个基于风险感知学习方法的综合框架，用于实现鲁棒的查询优化。

    

    关系数据库管理系统(RDBMS)中的查询优化器搜索预期对于给定查询最优的执行计划。它们使用参数估计，通常是不准确的，并且做出的假设在实践中可能不成立。因此，在这些估计和假设无效时，它们可能选择在运行时是次优的执行计划，这可能导致查询性能不佳。因此，查询优化器不足以支持鲁棒的查询优化。近年来，使用机器学习(ML)来提高数据系统的效率并减少其维护开销的兴趣日益高涨，在查询优化领域取得了有希望的结果。在本文中，受到这些进展的启发，并基于IBM Db2多年的经验，我们提出了Roq: 一种基于风险感知学习方法的综合框架，它实现了鲁棒的查询优化。

    Query optimizers in relational database management systems (RDBMSs) search for execution plans expected to be optimal for a given queries. They use parameter estimates, often inaccurate, and make assumptions that may not hold in practice. Consequently, they may select execution plans that are suboptimal at runtime, when these estimates and assumptions are not valid, which may result in poor query performance. Therefore, query optimizers do not sufficiently support robust query optimization. Recent years have seen a surge of interest in using machine learning (ML) to improve efficiency of data systems and reduce their maintenance overheads, with promising results obtained in the area of query optimization in particular. In this paper, inspired by these advancements, and based on several years of experience of IBM Db2 in this journey, we propose Robust Optimization of Queries, (Roq), a holistic framework that enables robust query optimization based on a risk-aware learning approach. Roq 
    

