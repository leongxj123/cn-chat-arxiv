# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Active Learning of Mealy Machines with Timers](https://arxiv.org/abs/2403.02019) | 这篇论文提出了一种用于查询学习具有定时器的Mealy机器的算法，在实现上明显比已有算法更有效率。 |
| [^2] | [Masked Hard-Attention Transformers and Boolean RASP Recognize Exactly the Star-Free Languages.](http://arxiv.org/abs/2310.13897) | 给出了一种新的变换器编码器模型，该模型具有硬注意力和严格未来掩码，并且证明这些网络识别的语言类别正是无星语言。研究还发现，通过添加位置嵌入，这一模型可以扩展到其他研究充分的语言类别。一个关键技术是布尔RASP，通过无星语言的研究，将变换器与一阶逻辑、时态逻辑和代数自动机理论相关联。 |

# 详细

[^1]: 具有定时器的Mealy机器的主动学习

    Active Learning of Mealy Machines with Timers

    [https://arxiv.org/abs/2403.02019](https://arxiv.org/abs/2403.02019)

    这篇论文提出了一种用于查询学习具有定时器的Mealy机器的算法，在实现上明显比已有算法更有效率。

    

    我们在黑盒环境中提出了第一个用于查询学习一般类别的具有定时器的Mealy机器（MMTs）的算法。我们的算法是Vaandrager等人的L＃算法对定时设置的扩展。类似于Waga提出的用于学习定时自动机的算法，我们的算法受到Maler＆Pnueli思想的启发。我们的算法和Waga的算法都使用符号查询进行基础语言学习，然后使用有限数量的具体查询进行实现。然而，Waga需要指数级的具体查询来实现单个符号查询，而我们只需要多项式数量。这是因为要学习定时自动机，学习者需要确定每个转换的确切卫兵和重置（有指数多种可能性），而要学习MMT，学习者只需要弄清楚哪些先前的转换导致超时。正如我们之前的工作所示，

    arXiv:2403.02019v1 Announce Type: cross  Abstract: We present the first algorithm for query learning of a general class of Mealy machines with timers (MMTs) in a black-box context. Our algorithm is an extension of the L# algorithm of Vaandrager et al. to a timed setting. Like the algorithm for learning timed automata proposed by Waga, our algorithm is inspired by ideas of Maler & Pnueli. Based on the elementary languages of, both Waga's and our algorithm use symbolic queries, which are then implemented using finitely many concrete queries. However, whereas Waga needs exponentially many concrete queries to implement a single symbolic query, we only need a polynomial number. This is because in order to learn a timed automaton, a learner needs to determine the exact guard and reset for each transition (out of exponentially many possibilities), whereas for learning an MMT a learner only needs to figure out which of the preceding transitions caused a timeout. As shown in our previous work, 
    
[^2]: 掩码硬注意力变换器和布尔RASP准确识别无星语言。

    Masked Hard-Attention Transformers and Boolean RASP Recognize Exactly the Star-Free Languages. (arXiv:2310.13897v2 [cs.FL] UPDATED)

    [http://arxiv.org/abs/2310.13897](http://arxiv.org/abs/2310.13897)

    给出了一种新的变换器编码器模型，该模型具有硬注意力和严格未来掩码，并且证明这些网络识别的语言类别正是无星语言。研究还发现，通过添加位置嵌入，这一模型可以扩展到其他研究充分的语言类别。一个关键技术是布尔RASP，通过无星语言的研究，将变换器与一阶逻辑、时态逻辑和代数自动机理论相关联。

    

    我们考虑具有硬注意力（即所有注意力都集中在一个位置上）和严格的未来掩码（即每个位置只与严格左侧的位置进行注意力交互）的变换器编码器，并证明这些网络识别的语言类别正是无星语言。添加位置嵌入将被识别的语言类别扩展到其他研究充分的类别。这些证明中的一个关键技术是布尔RASP，它是一种受限于布尔值的RASP变种。通过无星语言，我们将变换器与一阶逻辑、时态逻辑和代数自动机理论联系起来。

    We consider transformer encoders with hard attention (in which all attention is focused on exactly one position) and strict future masking (in which each position only attends to positions strictly to its left), and prove that the class of languages recognized by these networks is exactly the star-free languages. Adding position embeddings increases the class of recognized languages to other well-studied classes. A key technique in these proofs is Boolean RASP, a variant of RASP that is restricted to Boolean values. Via the star-free languages, we relate transformers to first-order logic, temporal logic, and algebraic automata theory.
    

