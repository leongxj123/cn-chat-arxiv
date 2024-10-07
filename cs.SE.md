# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Data Augmentation for Code Translation with Comparable Corpora and Multiple References.](http://arxiv.org/abs/2311.00317) | 该论文介绍了两种数据增强方法来改善编程语言之间的代码翻译。通过构建可比较的语料库和增加多个参考翻译，实验结果表明这些方法显著提高了CodeT5在Java、Python和C++之间的翻译准确性。 |
| [^2] | [Automated Bug Generation in the era of Large Language Models.](http://arxiv.org/abs/2310.02407) | 本论文探讨了在大型语言模型时代的自动缺陷生成问题，针对难以检测和难以修复的缺陷提出了解决方案，并分析了基于学习的技术中这两个目标的冲突。 |

# 详细

[^1]: 用可比较的语料和多个参考文献进行代码翻译的数据增强

    Data Augmentation for Code Translation with Comparable Corpora and Multiple References. (arXiv:2311.00317v1 [cs.CL])

    [http://arxiv.org/abs/2311.00317](http://arxiv.org/abs/2311.00317)

    该论文介绍了两种数据增强方法来改善编程语言之间的代码翻译。通过构建可比较的语料库和增加多个参考翻译，实验结果表明这些方法显著提高了CodeT5在Java、Python和C++之间的翻译准确性。

    

    在编程语言之间进行代码翻译的一个主要挑战是平行训练数据通常有限。为了克服这个挑战，我们提出了两种数据增强技术，一种是构建可比较的语料库（即具有类似功能的代码对），另一种是用多个参考翻译来增强现有的平行数据。具体而言，我们构建并分析了多种类型的可比较的语料库，包括使用代码生成模型从自然语言文档中生成的程序。此外，为了减少对单个参考翻译的过拟合，我们自动生成了可用平行数据的额外翻译参考，并通过单元测试对翻译进行筛选，从而增加了目标翻译的变化。实验证明，我们的数据增强技术显著提高了CodeT5在Java、Python和C++之间的翻译准确性（平均提升了7.5%的计算准确性（CA@1））。

    One major challenge of translating code between programming languages is that parallel training data is often limited. To overcome this challenge, we present two data augmentation techniques, one that builds comparable corpora (i.e., code pairs with similar functionality), and another that augments existing parallel data with multiple reference translations. Specifically, we build and analyze multiple types of comparable corpora, including programs generated from natural language documentation using a code generation model. Furthermore, to reduce overfitting to a single reference translation, we automatically generate additional translation references for available parallel data and filter the translations by unit tests, which increases variation in target translations. Experiments show that our data augmentation techniques significantly improve CodeT5 for translation between Java, Python, and C++ by an average of 7.5% Computational Accuracy (CA@1), which verifies the correctness of tr
    
[^2]: 在大型语言模型时代的自动缺陷生成

    Automated Bug Generation in the era of Large Language Models. (arXiv:2310.02407v1 [cs.SE])

    [http://arxiv.org/abs/2310.02407](http://arxiv.org/abs/2310.02407)

    本论文探讨了在大型语言模型时代的自动缺陷生成问题，针对难以检测和难以修复的缺陷提出了解决方案，并分析了基于学习的技术中这两个目标的冲突。

    

    缺陷在软件工程中是至关重要的；过去几十年的许多研究已经提出了检测、定位和修复软件系统中的缺陷的方法。评估这些技术的有效性需要复杂的缺陷，即那些很难通过测试和调试来检测和修复的缺陷。从传统软件工程的角度来看，难以修复的缺陷与正确的代码在多个位置上有所差异，这使得它们难以定位和修复。而难以检测的缺陷则在特定的测试输入和可达条件下展现出来。这两个目标，即生成难以检测和难以修复的缺陷，大多数是一致的；缺陷生成技术可以将多个语句更改为仅在特定输入集合下被覆盖。然而，对于基于学习的技术来说，这两个目标是相互冲突的：一个缺陷应该有与训练数据中的正确代码相似的代码表示，以挑战缺陷预测。

    Bugs are essential in software engineering; many research studies in the past decades have been proposed to detect, localize, and repair bugs in software systems. Effectiveness evaluation of such techniques requires complex bugs, i.e., those that are hard to detect through testing and hard to repair through debugging. From the classic software engineering point of view, a hard-to-repair bug differs from the correct code in multiple locations, making it hard to localize and repair. Hard-to-detect bugs, on the other hand, manifest themselves under specific test inputs and reachability conditions. These two objectives, i.e., generating hard-to-detect and hard-to-repair bugs, are mostly aligned; a bug generation technique can change multiple statements to be covered only under a specific set of inputs. However, these two objectives are conflicting for learning-based techniques: A bug should have a similar code representation to the correct code in the training data to challenge a bug predi
    

