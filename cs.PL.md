# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CoverUp: Coverage-Guided LLM-Based Test Generation](https://arxiv.org/abs/2403.16218) | CoverUp通过覆盖率分析和大型语言模型相结合的方式，驱动生成高覆盖率的Python回归测试，并在改进覆盖率方面取得显著成就。 |

# 详细

[^1]: CoverUp：基于覆盖率引导的LLM测试生成系统

    CoverUp: Coverage-Guided LLM-Based Test Generation

    [https://arxiv.org/abs/2403.16218](https://arxiv.org/abs/2403.16218)

    CoverUp通过覆盖率分析和大型语言模型相结合的方式，驱动生成高覆盖率的Python回归测试，并在改进覆盖率方面取得显著成就。

    

    本文介绍了CoverUp，这是一个新型系统，通过覆盖率分析和大型语言模型（LLM）的结合驱动生成高覆盖率的Python回归测试。CoverUp通过迭代改善覆盖率，将覆盖率分析与LLM对话交替进行，以便将注意力集中在尚未涵盖的代码行和分支上。最终的测试套件相比当前技术水平显著提高了覆盖率：与CodaMosa相比，一种混合LLM / 基于搜索的软件测试系统，CoverUp在各方面都大幅提高了覆盖率。以模块为基础，CoverUp实现了81%的中位线覆盖率（对比62%）、53%的分支覆盖率（对比35%）和78%的线+分支覆盖率（对比55%）。我们展示了CoverUp的迭代、覆盖率引导方法对其有效性至关重要，为其成功的近一半作出了贡献。

    arXiv:2403.16218v1 Announce Type: cross  Abstract: This paper presents CoverUp, a novel system that drives the generation of high-coverage Python regression tests via a combination of coverage analysis and large-language models (LLMs). CoverUp iteratively improves coverage, interleaving coverage analysis with dialogs with the LLM to focus its attention on as yet uncovered lines and branches. The resulting test suites significantly improve coverage over the current state of the art: compared to CodaMosa, a hybrid LLM / search-based software testing system, CoverUp substantially improves coverage across the board. On a per-module basis, CoverUp achieves median line coverage of 81% (vs. 62%), branch coverage of 53% (vs. 35%) and line+branch coverage of 78% (vs. 55%). We show that CoverUp's iterative, coverage-guided approach is crucial to its effectiveness, contributing to nearly half of its successes.
    

