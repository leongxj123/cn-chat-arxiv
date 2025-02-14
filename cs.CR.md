# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Vision-LLMs Can Fool Themselves with Self-Generated Typographic Attacks](https://arxiv.org/abs/2402.00626) | 这项研究深入研究了大规模视觉语言模型（LVLM）对于自动生成的排版攻击的易受攻击性，并引入了一种新的、更有效的自动生成的排版攻击方法，为此设计了一个独特的测试基准。通过使用该基准，研究发现排版攻击对LVLM构成了重大威胁。 |

# 详细

[^1]: Vision-LLMs通过自动生成的排版攻击可以自欺欺人

    Vision-LLMs Can Fool Themselves with Self-Generated Typographic Attacks

    [https://arxiv.org/abs/2402.00626](https://arxiv.org/abs/2402.00626)

    这项研究深入研究了大规模视觉语言模型（LVLM）对于自动生成的排版攻击的易受攻击性，并引入了一种新的、更有效的自动生成的排版攻击方法，为此设计了一个独特的测试基准。通过使用该基准，研究发现排版攻击对LVLM构成了重大威胁。

    

    最近，在大规模视觉语言模型（LVLM）方面取得了重大进展；这是一种利用大型预训练语言模型的全新类别的视觉语言模型。然而，LVLM对于涉及将误导性文本叠加到图像上的从排版攻击的容易受攻击性却没有研究。此外，先前的排版攻击依赖于从预定义类别集合中随机选择一个误导性类别。然而，随机选择的类别可能不是最有效的攻击类别。为了解决这些问题，我们首先引入了一种独特设计的新颖基准来测试LVLM对排版攻击的容易受攻击性。此外，我们介绍了一种新而更有效的排版攻击：自动生成的排版攻击。实际上，我们的方法通过简单地提示GPT-4V等模型利用其强大的语言能力推荐一种排版攻击来为给定的图像生成攻击。使用我们的新颖基准，我们发现排版攻击对LVLM构成了重大威胁。

    Recently, significant progress has been made on Large Vision-Language Models (LVLMs); a new class of VL models that make use of large pre-trained language models. Yet, their vulnerability to Typographic attacks, which involve superimposing misleading text onto an image remain unstudied. Furthermore, prior work typographic attacks rely on sampling a random misleading class from a predefined set of classes. However, the random chosen class might not be the most effective attack. To address these issues, we first introduce a novel benchmark uniquely designed to test LVLMs vulnerability to typographic attacks. Furthermore, we introduce a new and more effective typographic attack: Self-Generated typographic attacks. Indeed, our method, given an image, make use of the strong language capabilities of models like GPT-4V by simply prompting them to recommend a typographic attack. Using our novel benchmark, we uncover that typographic attacks represent a significant threat against LVLM(s). Furth
    

