# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LEACE: Perfect linear concept erasure in closed form.](http://arxiv.org/abs/2306.03819) | 本文介绍了一种闭合形式的方法LEACE，可在删除指定特征的同时尽可能少地改变表示，并可证明防止所有线性分类器检测到概念。作者用“概念擦除”这一新方法将其应用于大型语言模型，在测量语言模型对词性的依赖性和减少BERT嵌入中的性别偏差任务中得出良好表现。 |

# 详细

[^1]: LEACE：闭合形式中的完美线性概念擦除

    LEACE: Perfect linear concept erasure in closed form. (arXiv:2306.03819v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.03819](http://arxiv.org/abs/2306.03819)

    本文介绍了一种闭合形式的方法LEACE，可在删除指定特征的同时尽可能少地改变表示，并可证明防止所有线性分类器检测到概念。作者用“概念擦除”这一新方法将其应用于大型语言模型，在测量语言模型对词性的依赖性和减少BERT嵌入中的性别偏差任务中得出良好表现。

    

    概念擦除旨在从表征中删除指定的特征。它可以提高公平性（例如，防止分类器使用性别或种族）和可解释性（例如，删除概念以观察模型行为的变化）。我们引入了LEAst-squares概念擦除（LEACE），这是一种闭合形式的方法，可证明防止所有线性分类器检测到概念，同时尽可能地改变表示，如广泛类别的范数所测量的那样。我们使用名为“概念擦除”的新方法将LEACE应用于大型语言模型，擦除每个层中的目标概念信息。我们在两个任务上展示了我们的方法：测量语言模型对词性信息的依赖性，以及减少BERT嵌入中的性别偏差。代码可在https://github.com/EleutherAI/concept-erasure上找到。

    Concept erasure aims to remove specified features from a representation. It can improve fairness (e.g. preventing a classifier from using gender or race) and interpretability (e.g. removing a concept to observe changes in model behavior). We introduce LEAst-squares Concept Erasure (LEACE), a closed-form method which provably prevents all linear classifiers from detecting a concept while changing the representation as little as possible, as measured by a broad class of norms. We apply LEACE to large language models with a novel procedure called "concept scrubbing," which erases target concept information from every layer in the network. We demonstrate our method on two tasks: measuring the reliance of language models on part-of-speech information, and reducing gender bias in BERT embeddings. Code is available at https://github.com/EleutherAI/concept-erasure.
    

