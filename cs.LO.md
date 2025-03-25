# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Notation3 as an Existential Rule Language.](http://arxiv.org/abs/2308.07332) | 本文研究了Notation3与存在规则之间的关系，并提出了一个将部分Notation3直接映射到存在规则的方法，从而提高了Notation3推理的效率。 |

# 详细

[^1]: Notation3作为一种存在规则语言

    Notation3 as an Existential Rule Language. (arXiv:2308.07332v1 [cs.AI])

    [http://arxiv.org/abs/2308.07332](http://arxiv.org/abs/2308.07332)

    本文研究了Notation3与存在规则之间的关系，并提出了一个将部分Notation3直接映射到存在规则的方法，从而提高了Notation3推理的效率。

    

    Notation3逻辑（\nthree）是RDF的扩展，允许用户编写引入新的空白节点到RDF图中的规则。许多应用程序（例如本体映射）依赖于此功能，因为空白节点在Web上广泛存在，直接使用或作为辅助结构。然而，涵盖该逻辑非常重要功能的快速\nthree推理器的数量相对有限。另一方面，像VLog或Nemo之类的引擎不直接支持语义Web规则格式，但是它们是为非常相似的构造（存在规则）开发和优化的。在本文中，我们研究了具有空白节点的\nthree规则与存在规则之间的关系。我们确定了一个可以直接映射到存在规则的\nthree子集，并定义了这样一个映射，保持了\nthree公式的等价性。为了进一步说明在某些情况下\nthree推理可以受益于我们的转换，我们使用该映射进行分析。

    Notation3 Logic (\nthree) is an extension of RDF that allows the user to write rules introducing new blank nodes to RDF graphs. Many applications (e.g., ontology mapping) rely on this feature as blank nodes -- used directly or in auxiliary constructs -- are omnipresent on the Web. However, the number of fast \nthree reasoners covering this very important feature of the logic is rather limited. On the other hand, there are engines like VLog or Nemo which do not directly support Semantic Web rule formats but which are developed and optimized for very similar constructs: existential rules. In this paper, we investigate the relation between \nthree rules with blank nodes in their heads and existential rules. We identify a subset of \nthree which can be mapped directly to existential rules and define such a mapping preserving the equivalence of \nthree formulae. In order to also illustrate that in some cases \nthree reasoning could benefit from our translation, we then employ this mapping i
    

