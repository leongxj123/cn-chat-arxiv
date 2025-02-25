# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ML-SUPERB: Multilingual Speech Universal PERformance Benchmark.](http://arxiv.org/abs/2305.10615) | 本文提出了一个覆盖143种语言、用于自我监督学习模型性能基准的多语种语音基准 ML-SUPERB，并发现自我监督学习模型可以显著提高性能且多语种模型不总是比单语言模型表现更好。 |

# 详细

[^1]: ML-SUPERB: 多语种语音自我监督学习性能基准

    ML-SUPERB: Multilingual Speech Universal PERformance Benchmark. (arXiv:2305.10615v1 [cs.SD])

    [http://arxiv.org/abs/2305.10615](http://arxiv.org/abs/2305.10615)

    本文提出了一个覆盖143种语言、用于自我监督学习模型性能基准的多语种语音基准 ML-SUPERB，并发现自我监督学习模型可以显著提高性能且多语种模型不总是比单语言模型表现更好。

    

    语音处理Universal PERformance Benchmark (SUPERB)是一个用于各种语音处理任务的自我监督学习模型性能基准的排行榜。然而，SUPERB在评估中主要考虑英语。本文介绍了多语种SUPERB (ML-SUPERB)，覆盖了143种语言（从高资源到濒危语言），考虑了自动语音识别和语言识别。与SUPERB概念类似，ML-SUPERB利用冻结的自我监督学习特征，并通过学习浅层下游模型的简单框架，用于多语种任务。与SUPERB基准类似，我们发现语音自我监督学习模型可以显著提高性能，与FBANK特征相比。此外，我们发现多语种模型并不总是比单语言模型表现更好。我们将发布ML-SUPERB作为一个挑战，提供组织好的数据集和可重现的训练脚本，用于未来的多语种表示研究。

    Speech processing Universal PERformance Benchmark (SUPERB) is a leaderboard to benchmark the performance of Self-Supervised Learning (SSL) models on various speech processing tasks. However, SUPERB largely considers English speech in its evaluation. This paper presents multilingual SUPERB (ML-SUPERB), covering 143 languages (ranging from high-resource to endangered), and considering both automatic speech recognition and language identification. Following the concept of SUPERB, ML-SUPERB utilizes frozen SSL features and employs a simple framework for multilingual tasks by learning a shallow downstream model. Similar to the SUPERB benchmark, we find speech SSL models can significantly improve performance compared to FBANK features. Furthermore, we find that multilingual models do not always perform better than their monolingual counterparts. We will release ML-SUPERB as a challenge with organized datasets and reproducible training scripts for future multilingual representation research.
    

