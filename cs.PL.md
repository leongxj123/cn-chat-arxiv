# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Amortizing Pragmatic Program Synthesis with Rankings.](http://arxiv.org/abs/2309.03225) | 该研究提出了一种使用全局实际排名方法来分摊实际程序合成中计算负担的方法，并证明了该方法适用于使用单个演示的实际合成器，并且通过实证研究展示了全局排名有效近似了全体合理响应。 |

# 详细

[^1]: 使用排名来摊销实际程序合成的成本的研究

    Amortizing Pragmatic Program Synthesis with Rankings. (arXiv:2309.03225v1 [cs.PL])

    [http://arxiv.org/abs/2309.03225](http://arxiv.org/abs/2309.03225)

    该研究提出了一种使用全局实际排名方法来分摊实际程序合成中计算负担的方法，并证明了该方法适用于使用单个演示的实际合成器，并且通过实证研究展示了全局排名有效近似了全体合理响应。

    

    在程序合成中，一个智能系统接收一组用户生成的示例，并返回一个逻辑一致的程序。使用合理演说行为（RSA）框架在构建“实际”程序合成器方面取得了成功，这些合成器返回的程序不仅在逻辑上一致，而且还考虑了用户如何选择示例。然而，运行RSA算法的计算负担限制了实际程序合成在可能程序个数较少的领域的应用。本研究提出了一种通过利用“全局实际排名” - 一个单一的、总的排列所有假设的方法来分摊RSA算法的计算负担。我们证明，在使用单个演示的实际合成器中，我们的全局排名方法完全复制了RSA的排序响应。我们进一步通过实证研究显示，全局排名有效地近似了全体合理响应。

    In program synthesis, an intelligent system takes in a set of user-generated examples and returns a program that is logically consistent with these examples. The usage of Rational Speech Acts (RSA) framework has been successful in building \emph{pragmatic} program synthesizers that return programs which -in addition to being logically consistent -- account for the fact that a user chooses their examples informatively. However, the computational burden of running the RSA algorithm has restricted the application of pragmatic program synthesis to domains with a small number of possible programs. This work presents a novel method of amortizing the RSA algorithm by leveraging a \emph{global pragmatic ranking} -- a single, total ordering of all the hypotheses. We prove that for a pragmatic synthesizer that uses a single demonstration, our global ranking method exactly replicates RSA's ranked responses. We further empirically show that global rankings effectively approximate the full pragma
    

