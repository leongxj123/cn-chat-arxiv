# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Generative AI Paradox on Evaluation: What It Can Solve, It May Not Evaluate](https://arxiv.org/abs/2402.06204) | 本文讨论了生成AI评估中的悖论，并发现了大型语言模型在评估任务中性能较差的现象。研究突出了需要检查模型作为评估者的忠实度和可信度，以及探索生成优秀与评估能力之间的关联。 |

# 详细

[^1]: 生成AI评估中的悖论：它能解决的问题可能无法进行评估

    The Generative AI Paradox on Evaluation: What It Can Solve, It May Not Evaluate

    [https://arxiv.org/abs/2402.06204](https://arxiv.org/abs/2402.06204)

    本文讨论了生成AI评估中的悖论，并发现了大型语言模型在评估任务中性能较差的现象。研究突出了需要检查模型作为评估者的忠实度和可信度，以及探索生成优秀与评估能力之间的关联。

    

    本文探讨了一种假设，即在生成任务中擅长的大型语言模型（LLM）同样擅长作为评估者。我们使用TriviaQA数据集评估了三个LLM和一个开源LM在问答（QA）和评估任务中的表现。结果表明存在显著差异，LLM在评估任务中的性能较生成任务低。有趣的是，我们发现了一些不忠实的评估情况，模型在其不擅长的领域中准确评估答案，突出了需要检查LLM作为评估者的忠实度和可信度。本研究有助于理解“生成AI悖论”，强调了探索生成优秀与评估能力之间的关联以及审查模型评估中忠实度方面的必要性。

    This paper explores the assumption that Large Language Models (LLMs) skilled in generation tasks are equally adept as evaluators. We assess the performance of three LLMs and one open-source LM in Question-Answering (QA) and evaluation tasks using the TriviaQA (Joshi et al., 2017) dataset. Results indicate a significant disparity, with LLMs exhibiting lower performance in evaluation tasks compared to generation tasks. Intriguingly, we discover instances of unfaithful evaluation where models accurately evaluate answers in areas where they lack competence, underscoring the need to examine the faithfulness and trustworthiness of LLMs as evaluators. This study contributes to the understanding of "the Generative AI Paradox" (West et al., 2023), highlighting a need to explore the correlation between generative excellence and evaluation proficiency, and the necessity to scrutinize the faithfulness aspect in model evaluations.
    

