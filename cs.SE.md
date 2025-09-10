# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Automated Bug Generation in the era of Large Language Models.](http://arxiv.org/abs/2310.02407) | 本论文探讨了在大型语言模型时代的自动缺陷生成问题，针对难以检测和难以修复的缺陷提出了解决方案，并分析了基于学习的技术中这两个目标的冲突。 |

# 详细

[^1]: 在大型语言模型时代的自动缺陷生成

    Automated Bug Generation in the era of Large Language Models. (arXiv:2310.02407v1 [cs.SE])

    [http://arxiv.org/abs/2310.02407](http://arxiv.org/abs/2310.02407)

    本论文探讨了在大型语言模型时代的自动缺陷生成问题，针对难以检测和难以修复的缺陷提出了解决方案，并分析了基于学习的技术中这两个目标的冲突。

    

    缺陷在软件工程中是至关重要的；过去几十年的许多研究已经提出了检测、定位和修复软件系统中的缺陷的方法。评估这些技术的有效性需要复杂的缺陷，即那些很难通过测试和调试来检测和修复的缺陷。从传统软件工程的角度来看，难以修复的缺陷与正确的代码在多个位置上有所差异，这使得它们难以定位和修复。而难以检测的缺陷则在特定的测试输入和可达条件下展现出来。这两个目标，即生成难以检测和难以修复的缺陷，大多数是一致的；缺陷生成技术可以将多个语句更改为仅在特定输入集合下被覆盖。然而，对于基于学习的技术来说，这两个目标是相互冲突的：一个缺陷应该有与训练数据中的正确代码相似的代码表示，以挑战缺陷预测。

    Bugs are essential in software engineering; many research studies in the past decades have been proposed to detect, localize, and repair bugs in software systems. Effectiveness evaluation of such techniques requires complex bugs, i.e., those that are hard to detect through testing and hard to repair through debugging. From the classic software engineering point of view, a hard-to-repair bug differs from the correct code in multiple locations, making it hard to localize and repair. Hard-to-detect bugs, on the other hand, manifest themselves under specific test inputs and reachability conditions. These two objectives, i.e., generating hard-to-detect and hard-to-repair bugs, are mostly aligned; a bug generation technique can change multiple statements to be covered only under a specific set of inputs. However, these two objectives are conflicting for learning-based techniques: A bug should have a similar code representation to the correct code in the training data to challenge a bug predi
    

