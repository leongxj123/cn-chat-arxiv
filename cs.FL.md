# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MLRegTest: A Benchmark for the Machine Learning of Regular Languages.](http://arxiv.org/abs/2304.07687) | 本文提出了一个名为MLRegTest的新基准测试，其包含了来自1,800个正则语言的数据集。该测试根据逻辑复杂度和逻辑文字种类组织语言，并可以帮助我们了解机器学习系统在学习不同种类的长距离依赖方面的性能。 |

# 详细

[^1]: MLRegTest：机器学习正则语言的基准测试

    MLRegTest: A Benchmark for the Machine Learning of Regular Languages. (arXiv:2304.07687v1 [cs.LG])

    [http://arxiv.org/abs/2304.07687](http://arxiv.org/abs/2304.07687)

    本文提出了一个名为MLRegTest的新基准测试，其包含了来自1,800个正则语言的数据集。该测试根据逻辑复杂度和逻辑文字种类组织语言，并可以帮助我们了解机器学习系统在学习不同种类的长距离依赖方面的性能。

    

    评估机器学习系统对已知分类器的学习能力允许细致地检查它们可以学习哪些模式，并在将它们应用于未知分类器的学习时建立信心。本文提出了一个名为MLRegTest的新的序列分类机器学习系统基准测试，其中包含来自1,800个正则语言的训练、开发和测试集。不同类型的形式语言代表着不同种类的长距离依赖，并正确地识别序列中的长距离依赖是机器学习系统成功泛化的已知挑战。MLRegTest根据它们的逻辑复杂度（单调二阶，一阶，命题或单项式表达式）和逻辑文字的种类（字符串，定级字符串，子序列或两者的组合）组织其语言。逻辑复杂度和文字的选择提供了一种系统方法来理解不同种类的长距离依赖和机器学习系统在处理它们时的性能。

    Evaluating machine learning (ML) systems on their ability to learn known classifiers allows fine-grained examination of the patterns they can learn, which builds confidence when they are applied to the learning of unknown classifiers. This article presents a new benchmark for ML systems on sequence classification called MLRegTest, which contains training, development, and test sets from 1,800 regular languages.  Different kinds of formal languages represent different kinds of long-distance dependencies, and correctly identifying long-distance dependencies in sequences is a known challenge for ML systems to generalize successfully. MLRegTest organizes its languages according to their logical complexity (monadic second order, first order, propositional, or monomial expressions) and the kind of logical literals (string, tier-string, subsequence, or combinations thereof). The logical complexity and choice of literal provides a systematic way to understand different kinds of long-distance d
    

