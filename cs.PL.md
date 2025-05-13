# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LOOPer: A Learned Automatic Code Optimizer For Polyhedral Compilers](https://arxiv.org/abs/2403.11522) | LOOPer是针对多面体编译器的学习型自动代码优化器，通过机器学习建立成本模型来指导多面体优化搜索，突破了传统编译器在选择代码转换方面的限制。 |

# 详细

[^1]: LOOPer: 一个针对多面体编译器的学习型自动代码优化器

    LOOPer: A Learned Automatic Code Optimizer For Polyhedral Compilers

    [https://arxiv.org/abs/2403.11522](https://arxiv.org/abs/2403.11522)

    LOOPer是针对多面体编译器的学习型自动代码优化器，通过机器学习建立成本模型来指导多面体优化搜索，突破了传统编译器在选择代码转换方面的限制。

    

    虽然多面体编译器在实现高级代码转换方面已经取得成功，但在选择能够带来最佳加速的最有利转换方面仍然面临挑战。这促使使用机器学习构建成本模型来引导多面体优化的搜索。最先进的多面体编译器已经展示了这种方法的可行性概念验证。虽然这种概念验证显示出了希望，但仍然存在显著限制。使用深度学习成本模型的最先进多面体编译器只支持少量仿射变换的子集，限制了它们应用复杂代码变换的能力。它们还只支持具有单个循环嵌套和矩形迭代域的简单程序，限制了它们对许多程序的适用性。这些限制显著影响了这样的编译器和自动调度器的通用性

    arXiv:2403.11522v1 Announce Type: cross  Abstract: While polyhedral compilers have shown success in implementing advanced code transformations, they still have challenges in selecting the most profitable transformations that lead to the best speedups. This has motivated the use of machine learning to build cost models to guide the search for polyhedral optimizations. State-of-the-art polyhedral compilers have demonstrated a viable proof-of-concept of this approach. While such a proof-of-concept has shown promise, it still has significant limitations. State-of-the-art polyhedral compilers that use a deep-learning cost model only support a small subset of affine transformations, limiting their ability to apply complex code transformations. They also only support simple programs that have a single loop nest and a rectangular iteration domain, limiting their applicability to many programs. These limitations significantly impact the generality of such compilers and autoschedulers and put in
    

