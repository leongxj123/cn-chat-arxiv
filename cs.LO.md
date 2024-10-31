# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learnability is a Compact Property](https://arxiv.org/abs/2402.10360) | 监督学习问题的困难性具有紧凑的有限特性表征。 |
| [^2] | [Masked Hard-Attention Transformers and Boolean RASP Recognize Exactly the Star-Free Languages.](http://arxiv.org/abs/2310.13897) | 给出了一种新的变换器编码器模型，该模型具有硬注意力和严格未来掩码，并且证明这些网络识别的语言类别正是无星语言。研究还发现，通过添加位置嵌入，这一模型可以扩展到其他研究充分的语言类别。一个关键技术是布尔RASP，通过无星语言的研究，将变换器与一阶逻辑、时态逻辑和代数自动机理论相关联。 |
| [^3] | [Investigations into Proof Structures.](http://arxiv.org/abs/2304.12827) | 介绍了一种新的形式主义来操作和分析证明，用于生成更短的证明和减少搜索工作量。 |

# 详细

[^1]: 学习性是一种紧凑性质

    Learnability is a Compact Property

    [https://arxiv.org/abs/2402.10360](https://arxiv.org/abs/2402.10360)

    监督学习问题的困难性具有紧凑的有限特性表征。

    

    最近关于学习的工作取得了一个引人注目的结果：各种问题的可学习性可能是不可判定的，或者与标准集合论ZFC公理无关。此外，这种问题的可学习性可能不是具有有限特性的属性：非正式地说，它不能通过检查问题的有限投影来检测。

    arXiv:2402.10360v1 Announce Type: new  Abstract: Recent work on learning has yielded a striking result: the learnability of various problems can be undecidable, or independent of the standard ZFC axioms of set theory. Furthermore, the learnability of such problems can fail to be a property of finite character: informally, it cannot be detected by examining finite projections of the problem.   On the other hand, learning theory abounds with notions of dimension that characterize learning and consider only finite restrictions of the problem, i.e., are properties of finite character. How can these results be reconciled? More precisely, which classes of learning problems are vulnerable to logical undecidability, and which are within the grasp of finite characterizations?   We demonstrate that the difficulty of supervised learning with metric losses admits a tight finite characterization. In particular, we prove that the sample complexity of learning a hypothesis class can be detected by ex
    
[^2]: 掩码硬注意力变换器和布尔RASP准确识别无星语言。

    Masked Hard-Attention Transformers and Boolean RASP Recognize Exactly the Star-Free Languages. (arXiv:2310.13897v2 [cs.FL] UPDATED)

    [http://arxiv.org/abs/2310.13897](http://arxiv.org/abs/2310.13897)

    给出了一种新的变换器编码器模型，该模型具有硬注意力和严格未来掩码，并且证明这些网络识别的语言类别正是无星语言。研究还发现，通过添加位置嵌入，这一模型可以扩展到其他研究充分的语言类别。一个关键技术是布尔RASP，通过无星语言的研究，将变换器与一阶逻辑、时态逻辑和代数自动机理论相关联。

    

    我们考虑具有硬注意力（即所有注意力都集中在一个位置上）和严格的未来掩码（即每个位置只与严格左侧的位置进行注意力交互）的变换器编码器，并证明这些网络识别的语言类别正是无星语言。添加位置嵌入将被识别的语言类别扩展到其他研究充分的类别。这些证明中的一个关键技术是布尔RASP，它是一种受限于布尔值的RASP变种。通过无星语言，我们将变换器与一阶逻辑、时态逻辑和代数自动机理论联系起来。

    We consider transformer encoders with hard attention (in which all attention is focused on exactly one position) and strict future masking (in which each position only attends to positions strictly to its left), and prove that the class of languages recognized by these networks is exactly the star-free languages. Adding position embeddings increases the class of recognized languages to other well-studied classes. A key technique in these proofs is Boolean RASP, a variant of RASP that is restricted to Boolean values. Via the star-free languages, we relate transformers to first-order logic, temporal logic, and algebraic automata theory.
    
[^3]: 证明结构的研究

    Investigations into Proof Structures. (arXiv:2304.12827v1 [cs.AI])

    [http://arxiv.org/abs/2304.12827](http://arxiv.org/abs/2304.12827)

    介绍了一种新的形式主义来操作和分析证明，用于生成更短的证明和减少搜索工作量。

    

    我们引入并详细阐述了一种新型形式主义来操作和分析证明作为一个整体的对象。在这第一次尝试中，这个形式主义仅限于由浓缩推导特征的一阶问题。我们以一个全面的形式重构和分析历史上{\L}ukasiewicz广泛研究过的问题的证明为例进行了阐述。这种方法为在证明搜索过程中生成引理提供了新的系统方法，以减少搜索工作量并找到更短的证明。在这条路线上报告了许多实验，其中自动发现了一个证明{\L}ukasiewicz的问题，它比以前任何由人或机器发现的证明都要短得多。

    We introduce and elaborate a novel formalism for the manipulation and analysis of proofs as objects in a global manner. In this first approach the formalism is restricted to first-order problems characterized by condensed detachment. It is applied in an exemplary manner to a coherent and comprehensive formal reconstruction and analysis of historical proofs of a widely-studied problem due to {\L}ukasiewicz. The underlying approach opens the door towards new systematic ways of generating lemmas in the course of proof search to the effects of reducing the search effort and finding shorter proofs. Among the numerous reported experiments along this line, a proof of {\L}ukasiewicz's problem was automatically discovered that is much shorter than any proof found before by man or machine.
    

