# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Purely Regular Approach to Non-Regular Core Spanners](https://arxiv.org/abs/2010.13442) | 本文提出了一种纯粹规则的非规则核心跨度生成方法，通过将字符串相等选择直接纳入底层规则语言中，可以得到具有略微较弱表达能力的核心跨度生成器的片段。 |

# 详细

[^1]: 一种纯粹规则的非规则核心跨度生成方法

    A Purely Regular Approach to Non-Regular Core Spanners

    [https://arxiv.org/abs/2010.13442](https://arxiv.org/abs/2010.13442)

    本文提出了一种纯粹规则的非规则核心跨度生成方法，通过将字符串相等选择直接纳入底层规则语言中，可以得到具有略微较弱表达能力的核心跨度生成器的片段。

    

    规则跨度生成器是通过vset-自动机特征化的，它们对并集、连接和投影等代数操作封闭，并具有理想的算法属性。核心跨度生成器作为IBM SystemT中查询语言AQL的核心功能的形式化引入，除了需要字符串相等选择外，还被证明会导致静态分析和查询评估中典型问题的高复杂性甚至不可判定性。我们提出了一种替代性的核心跨度生成方法：将字符串相等选择直接纳入表示底层规则跨度生成器的规则语言中（而不是将其视为在规则跨度生成器提取的表上的代数操作），我们得到了一个具有略微较弱表达能力的核心跨度生成器的片段。

    The regular spanners (characterised by vset-automata) are closed under the algebraic operations of union, join and projection, and have desirable algorithmic properties. The core spanners (introduced by Fagin, Kimelfeld, Reiss, and Vansummeren (PODS 2013, JACM 2015) as a formalisation of the core functionality of the query language AQL used in IBM's SystemT) additionally need string-equality selections and it has been shown by Freydenberger and Holldack (ICDT 2016, Theory of Computing Systems 2018) that this leads to high complexity and even undecidability of the typical problems in static analysis and query evaluation. We propose an alternative approach to core spanners: by incorporating the string-equality selections directly into the regular language that represents the underlying regular spanner (instead of treating it as an algebraic operation on the table extracted by the regular spanner), we obtain a fragment of core spanners that, while having slightly weaker expressive power t
    

