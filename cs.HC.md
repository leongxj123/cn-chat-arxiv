# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [If in a Crowdsourced Data Annotation Pipeline, a GPT-4](https://arxiv.org/abs/2402.16795) | 本文比较了 GPT-4 和 MTurk 管道的数据标注准确性，发现尽管 MTurk 采用了最佳实践，但 GPT-4 的准确率更高，并且结合 GPT-4 和众包标签使用聚合算法可以提高准确率。 |
| [^2] | [Bringing Generative AI to Adaptive Learning in Education](https://arxiv.org/abs/2402.14601) | 生成式人工智能技术与自适应学习概念的交叉研究将对教育中下一阶段学习格式的发展做出重要贡献。 |
| [^3] | [Does Writing with Language Models Reduce Content Diversity?](https://arxiv.org/abs/2309.05196) | 写作时使用InstructGPT（而不是GPT3）会显著降低内容多样性，增加不同作者之间的相似性，并减少整体的词汇和内容多样性。 |
| [^4] | ['Team-in-the-loop' organisational oversight of high-stakes AI.](http://arxiv.org/abs/2303.14007) | 本论文通过对团队在 AI 系统中的监管流程的纵向观察，探讨了 AI 系统对临床决策制定中团队监管的影响，研究发现此前的专业团队监管方法主要依靠解释和问询来获取信息，而 AI 的引入将可能在信息披露和决策制定方面造成一定程度的影响。 |

# 详细

[^1]: 如果在一个众包数据标注管道中，GPT-4

    If in a Crowdsourced Data Annotation Pipeline, a GPT-4

    [https://arxiv.org/abs/2402.16795](https://arxiv.org/abs/2402.16795)

    本文比较了 GPT-4 和 MTurk 管道的数据标注准确性，发现尽管 MTurk 采用了最佳实践，但 GPT-4 的准确率更高，并且结合 GPT-4 和众包标签使用聚合算法可以提高准确率。

    

    最近的研究表明GPT-4在数据标注准确性方面优于在线众包工作者，尤其是来自亚马逊机械土耳其（MTurk）的工作者。然而，这些研究因偏离标准众包实践并强调个别工作者的表现而受到批评，而不是整个数据标注过程。本文比较了GPT-4和一个道德且执行良好的MTurk管道，使用415名工作者标注了来自200篇学术文章的3,177个句段，使用了CODA-19方案。两个工作者界面产生了127,080个标签，然后通过八种标签聚合算法推断出最终的标签。我们的评估结果显示，尽管采用了最佳实践，MTurk管道的最高准确率为81.5%，而GPT-4达到了83.6%。有趣的是，当将GPT-4的标签与通过先进工作者界面收集的众包标签结合起来进行聚合时，8种算法中有2种实现了更高的准确率。

    arXiv:2402.16795v1 Announce Type: cross  Abstract: Recent studies indicated GPT-4 outperforms online crowd workers in data labeling accuracy, notably workers from Amazon Mechanical Turk (MTurk). However, these studies were criticized for deviating from standard crowdsourcing practices and emphasizing individual workers' performances over the whole data-annotation process. This paper compared GPT-4 and an ethical and well-executed MTurk pipeline, with 415 workers labeling 3,177 sentence segments from 200 scholarly articles using the CODA-19 scheme. Two worker interfaces yielded 127,080 labels, which were then used to infer the final labels through eight label-aggregation algorithms. Our evaluation showed that despite best practices, MTurk pipeline's highest accuracy was 81.5%, whereas GPT-4 achieved 83.6%. Interestingly, when combining GPT-4's labels with crowd labels collected via an advanced worker interface for aggregation, 2 out of the 8 algorithms achieved an even higher accuracy (
    
[^2]: 将生成式人工智能引入教育中的自适应学习

    Bringing Generative AI to Adaptive Learning in Education

    [https://arxiv.org/abs/2402.14601](https://arxiv.org/abs/2402.14601)

    生成式人工智能技术与自适应学习概念的交叉研究将对教育中下一阶段学习格式的发展做出重要贡献。

    

    最近生成式人工智能技术的激增，如大型语言模型和扩散模型，推动了人工智能在科学、金融和教育等各个领域的应用发展。与此同时，自适应学习这一概念在教育领域引起了极大关注，并证明其在提高学生学习效率方面的有效性。在本立场论文中，我们旨在探讨将生成式人工智能与自适应学习概念结合起来的交叉研究。通过讨论这一领域的好处、挑战和潜力，我们认为这种结合将为教育中下一阶段学习形式的发展做出重要贡献。

    arXiv:2402.14601v1 Announce Type: cross  Abstract: The recent surge in generative AI technologies, such as large language models and diffusion models, have boosted the development of AI applications in various domains, including science, finance, and education. Concurrently, adaptive learning, a concept that has gained substantial interest in the educational sphere, has proven its efficacy in enhancing students' learning efficiency. In this position paper, we aim to shed light on the intersectional studies of these two methods, which combine generative AI with adaptive learning concepts. By presenting discussions about the benefits, challenges, and potentials in this field, we argue that this union will contribute significantly to the development of the next stage learning format in education.
    
[^3]: 语言模型写作是否会降低内容多样性？

    Does Writing with Language Models Reduce Content Diversity?

    [https://arxiv.org/abs/2309.05196](https://arxiv.org/abs/2309.05196)

    写作时使用InstructGPT（而不是GPT3）会显著降低内容多样性，增加不同作者之间的相似性，并减少整体的词汇和内容多样性。

    

    大型语言模型（LLMs）引发了与模型辅助合作写作的激增。当不同用户纳入同一模型的建议时，会存在内容多样性减少的风险，可能限制公共话语中的多元观点。本研究通过控制实验测量了协同写作对多样性的影响，在该实验中，用户以三种设置撰写议论性文章--使用基本LLM（GPT3）、经过反馈调整的LLM（InstructGPT）以及不使用模型帮助写作。我们开发了一组多样性指标，并发现使用InstructGPT进行写作（而不是GPT3）会导致多样性明显降低。具体而言，它增加了不同作者的写作之间的相似性，减少了整体的词汇和内容多样性。此外，我们还发现这种影响主要来源于InstructGPT对共同撰写的文本贡献较少。

    arXiv:2309.05196v2 Announce Type: replace  Abstract: Large language models (LLMs) have led to a surge in collaborative writing with model assistance. As different users incorporate suggestions from the same model, there is a risk of decreased diversity in the produced content, potentially limiting diverse perspectives in public discourse. In this work, we measure the impact of co-writing on diversity via a controlled experiment, where users write argumentative essays in three setups -- using a base LLM (GPT3), a feedback-tuned LLM (InstructGPT), and writing without model help. We develop a set of diversity metrics and find that writing with InstructGPT (but not the GPT3) results in a statistically significant reduction in diversity. Specifically, it increases the similarity between the writings of different authors and reduces the overall lexical and content diversity. We additionally find that this effect is mainly attributable to InstructGPT contributing less diverse text to co-writt
    
[^4]: 高风险 AI 的团队监管：团队在循环中

    'Team-in-the-loop' organisational oversight of high-stakes AI. (arXiv:2303.14007v1 [cs.CY])

    [http://arxiv.org/abs/2303.14007](http://arxiv.org/abs/2303.14007)

    本论文通过对团队在 AI 系统中的监管流程的纵向观察，探讨了 AI 系统对临床决策制定中团队监管的影响，研究发现此前的专业团队监管方法主要依靠解释和问询来获取信息，而 AI 的引入将可能在信息披露和决策制定方面造成一定程度的影响。

    

    监管对于高风险公共部门 AI 应用程序至关重要，因为决策可能会对个人和集体产生深远影响。目前在公共部门中关于 AI 监管机制的许多思考都围绕着人类决策者处于 "循环中 "这一概念，并且能够干预以防止错误和潜在危害。然而，在许多高风险公共部门背景下，决策的运营监管是由专业团队而不是个人进行的。部署的 AI 系统如何整合到这些现有的团队监管流程中，尚未引起太多注意。我们通过制度分析探讨 AI 对临床决策制定的现有监管的影响，填补该方面的空白。我们发现，现有的监管嵌套在专业培训要求中，并且在征询关键信息时 heavilyrely  于解释和提问。专业团队使用各种会计披露技术来警告同事和监管行为。我们考虑了在 AI 系统引入到现有的团队监管流程中，信息披露和决策制定可能发生改变的几种方式。

    Oversight is rightly recognised as vital within high-stakes public sector AI applications, where decisions can have profound individual and collective impacts. Much current thinking regarding forms of oversight mechanisms for AI within the public sector revolves around the idea of human decision makers being 'in-the-loop' and thus being able to intervene to prevent errors and potential harm. However, in a number of high-stakes public sector contexts, operational oversight of decisions is made by expert teams rather than individuals. The ways in which deployed AI systems can be integrated into these existing operational team oversight processes has yet to attract much attention. We address this gap by exploring the impacts of AI upon pre-existing oversight of clinical decision-making through institutional analysis. We find that existing oversight is nested within professional training requirements and relies heavily upon explanation and questioning to elicit vital information. Professio
    

