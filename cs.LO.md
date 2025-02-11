# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LOCALINTEL: Generating Organizational Threat Intelligence from Global and Local Cyber Knowledge.](http://arxiv.org/abs/2401.10036) | LOCALINTEL是一个自动化的知识上下文化系统，利用大型语言模型的能力，从全球和本地知识数据库中自动生成组织的威胁情报。 |
| [^2] | [Semiring Provenance for Lightweight Description Logics.](http://arxiv.org/abs/2310.16472) | 这篇论文研究了在描述逻辑中使用半环溯源的框架，并定义了一种适用于轻量级描述逻辑的溯源语义。论文证明了在半环施加限制的情况下，语义满足一些重要的特性，并对why溯源方法进行了研究。 |

# 详细

[^1]: LOCALINTEL：从全球和本地网络知识生成组织威胁情报

    LOCALINTEL: Generating Organizational Threat Intelligence from Global and Local Cyber Knowledge. (arXiv:2401.10036v1 [cs.CR])

    [http://arxiv.org/abs/2401.10036](http://arxiv.org/abs/2401.10036)

    LOCALINTEL是一个自动化的知识上下文化系统，利用大型语言模型的能力，从全球和本地知识数据库中自动生成组织的威胁情报。

    

    安全操作中心（SoC）分析师从公开访问的全球威胁数据库中收集威胁报告，并手动自定义以适应特定组织的需求。这些分析师还依赖于内部存储库，作为组织的私有本地知识数据库。可信的网络情报、关键操作细节和相关组织信息都存储在这些本地知识数据库中。分析师利用这些全球和本地知识数据库从事一项繁重的任务，手动创建组织独特的威胁响应和缓解策略。最近，大型语言模型（LLMs）已经展示了高效处理大规模多样化知识源的能力。我们利用这种能力来处理全球和本地知识数据库，自动化生成组织特定的威胁情报。在这项工作中，我们提出了LOCALINTEL，这是一个新颖的自动化知识上下文化系统，可以从全球和本地知识数据库中生成组织的威胁情报。

    Security Operations Center (SoC) analysts gather threat reports from openly accessible global threat databases and customize them manually to suit a particular organization's needs. These analysts also depend on internal repositories, which act as private local knowledge database for an organization. Credible cyber intelligence, critical operational details, and relevant organizational information are all stored in these local knowledge databases. Analysts undertake a labor intensive task utilizing these global and local knowledge databases to manually create organization's unique threat response and mitigation strategies. Recently, Large Language Models (LLMs) have shown the capability to efficiently process large diverse knowledge sources. We leverage this ability to process global and local knowledge databases to automate the generation of organization-specific threat intelligence.  In this work, we present LOCALINTEL, a novel automated knowledge contextualization system that, upon 
    
[^2]: 适用于轻量级描述逻辑的半环溯源

    Semiring Provenance for Lightweight Description Logics. (arXiv:2310.16472v1 [cs.LO])

    [http://arxiv.org/abs/2310.16472](http://arxiv.org/abs/2310.16472)

    这篇论文研究了在描述逻辑中使用半环溯源的框架，并定义了一种适用于轻量级描述逻辑的溯源语义。论文证明了在半环施加限制的情况下，语义满足一些重要的特性，并对why溯源方法进行了研究。

    

    我们研究了半环溯源——一种最初在关系数据库环境中定义的成功框架，用于描述逻辑。在此上下文中，本体公理被用交换半环的元素进行注释，并且这些注释根据它们的推导方式传播到本体的结果中。我们定义了一种溯源语义，适用于包括几种轻量级描述逻辑的语言，并展示了它与为带有特定类型注释（如模糊度）的本体定义的其他语义之间的关系。我们证明了在一些对半环施加限制的情况下，语义满足一些期望的特性（如扩展了数据库中定义的半环溯源）。然后我们专注于著名的why溯源方法，它允许计算每个加法幂等和乘法幂等的交换半环的半环溯源，并研究了与这种溯源方法相关的问题的复杂性。

    We investigate semiring provenance--a successful framework originally defined in the relational database setting--for description logics. In this context, the ontology axioms are annotated with elements of a commutative semiring and these annotations are propagated to the ontology consequences in a way that reflects how they are derived. We define a provenance semantics for a language that encompasses several lightweight description logics and show its relationships with semantics that have been defined for ontologies annotated with a specific kind of annotation (such as fuzzy degrees). We show that under some restrictions on the semiring, the semantics satisfies desirable properties (such as extending the semiring provenance defined for databases). We then focus on the well-known why-provenance, which allows to compute the semiring provenance for every additively and multiplicatively idempotent commutative semiring, and for which we study the complexity of problems related to the prov
    

