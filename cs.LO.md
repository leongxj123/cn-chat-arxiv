# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CatE: Embedding $\mathcal{ALC}$ ontologies using category-theoretical semantics.](http://arxiv.org/abs/2305.07163) | 本文介绍了一种使用范畴论语义的方法CatE，该方法可以高效地将本体公理投影到图形中，并生成本体论的嵌入，从而在知识图谱的链接预测任务上取得更好的表现。 |

# 详细

[^1]: CatE：使用范畴论语义嵌入$\mathcal{ALC}$本体论

    CatE: Embedding $\mathcal{ALC}$ ontologies using category-theoretical semantics. (arXiv:2305.07163v1 [cs.LO])

    [http://arxiv.org/abs/2305.07163](http://arxiv.org/abs/2305.07163)

    本文介绍了一种使用范畴论语义的方法CatE，该方法可以高效地将本体公理投影到图形中，并生成本体论的嵌入，从而在知识图谱的链接预测任务上取得更好的表现。

    

    与语义Web本体一起进行机器学习遵循几个策略之一，其中包括将本体投影到图形结构中，并将图形嵌入或基于图形的机器学习方法应用于生成的图形。已经开发出了几种将本体公理投影到图形中的方法。然而，这些方法在它们可以投影的公理类型（全面性）、它们是否可逆（单射性）以及它们如何利用语义信息方面存在局限性。这些限制限制了它们可以应用的任务类型。逻辑语言的范畴论语义以范畴而不是集合的形式形式化解释，并且范畴具有类似于图形的结构。我们开发了CatE，它使用$\mathcal{ALC}$描述逻辑的范畴论公式来生成本体公理的图形表示。CatE投影具有全面性和单射性，因此克服了其他基于图形本体论的方法的限制。CatE还通过将$\mathcal{ALC}$的语法编码为一种范畴来利用语义信息，这使我们能够为本体论生成嵌入。我们在知识图谱的链接预测任务上评估了CatE，并表明它优于现有方法。

    Machine learning with Semantic Web ontologies follows several strategies, one of which involves projecting ontologies into graph structures and applying graph embeddings or graph-based machine learning methods to the resulting graphs. Several methods have been developed that project ontology axioms into graphs. However, these methods are limited in the type of axioms they can project (totality), whether they are invertible (injectivity), and how they exploit semantic information. These limitations restrict the kind of tasks to which they can be applied. Category-theoretical semantics of logic languages formalizes interpretations using categories instead of sets, and categories have a graph-like structure. We developed CatE, which uses the category-theoretical formulation of the semantics of the Description Logic $\mathcal{ALC}$ to generate a graph representation for ontology axioms. The CatE projection is total and injective, and therefore overcomes limitations of other graph-based ont
    

