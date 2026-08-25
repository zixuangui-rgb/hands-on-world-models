# 5.1　为什么是 JEPA：从世界模型到预测表征

探索机器智能的边界，一直是人工智能研究者长期追求的目标。

机器非常擅长计算，拥有远超人类的计算能力。随着以 ChatGPT 为代表的大语言模型兴起，机器在目标、规则和评价标准较为明确的任务上展现出了强大执行能力，并且这种能力还在以非常快的速度提升。

但是，即便机器智能在一些特定的任务上拥有非常强的表现，其局限性依然非常明显。

现实世界并不是一道已经出好的题。环境不会告诉机器什么重要、规则是什么，也不会为每个行为即时评分。机器若要自主行动，就必须自己理解环境、预测后果，并随变化调整计划。对人类和动物来说，这就像是天生的能力，但是对于机器而言，这并不简单。

Yann LeCun 认为，这是机器走向自主智能必须补上的关键缺口。2022 年，他在 [_A Path Towards Autonomous Machine Intelligence_](https://openreview.net/forum?id=BZ5a1r-kVsf) 中系统提出了一套以世界模型为核心的智能架构，并将 JEPA 作为其中的重要组成部分。

**JEPA 不是一个完整的智能系统，而是一种学习内部世界状态的架构思想。它不要求模型还原观测的全部细节，而是在表示空间中进行预测，从而保留对理解和预测世界变化有用的信息。它试图回答的是：机器能否从感知经验中自主学到一种能够描述环境、预测变化，并最终支持规划与行动的内部状态？**

## 5.1.1　从完整架构出发：机器究竟要预测什么？

### 一套完整的自主智能架构

LeCun 在文章中画出了一套自主智能体的完整蓝图，并将 JEPA 放进其中，作为学习世界模型的一种方案。

这套架构要形成一个闭环：**感知现在 → 预想未来 → 评价结果 → 选择行动 → 再次感知。**

JEPA 主要处理其中的“如何表示和预测世界”，并不独自决定目标或动作。

![LeCun 2022 年提出的自主机器智能架构](/jepa/lecun-2022-autonomous-intelligence-architecture.png)

_图 5.1　LeCun 提出的自主机器智能架构。图中 Cost 由 Intrinsic Cost 与 Critic 两部分组成；为保持图面清晰，Configurator 从其他模块接收信息的箭头没有画出。来源：[LeCun, 2022, Figure 2](https://openreview.net/forum?id=BZ5a1r-kVsf)；六个模块的简要说明也可参见 [Meta 官方介绍](https://ai.meta.com/blog/yann-lecun-advances-in-ai-research/)。_

整套架构包含六个模块。每个模块只回答一个问题：

| 模块                  | 作用                                           | 它回答的问题       |
| --------------------- | ---------------------------------------------- | ------------------ |
| **Perception**        | 从传感器输入中估计当前状态                     | 现在发生了什么？   |
| **World Model**       | 补全缺失信息，并预测自然演化或动作造成的未来   | 接下来可能怎样？   |
| **Cost**              | 用固有代价与可学习的 Critic 评价状态           | 哪种结果更好？     |
| **Actor**             | 提出并优化候选动作序列                         | 接下来怎么做？     |
| **Short-term Memory** | 保存过去、当前与预测的状态及其代价             | 刚才发生了什么？   |
| **Configurator**      | 根据当前任务调整感知、世界模型、代价与动作模块 | 此刻应该关注什么？ |

它的工作方式并不复杂。Actor 先提出几组候选动作；World Model 分别预演这些动作可能带来的后果；Cost 比较这些后果；Actor 执行当前看来最合适的一步。新的观测到来后，系统重新估计状态、重新预测、重新规划。

这张图是一份**研究蓝图**，不是已经完成的系统。论文明确留下了许多尚未解决的问题，例如怎样训练 Configurator、Critic 和长期可用的世界模型。

### 为什么“预测”是必要的？

在这套架构的规划路径中，Perception 只能告诉机器“现在怎样”，Cost 只能评价“这个状态好不好”，Actor 只能提出“可以怎么做”。真正把动作与后果连接起来的，是 World Model：

```text
候选动作  →  World Model  →  可能的未来状态  →  Cost
   ↑                                                │
   └──────────── 选择预测代价更低的动作 ────────────┘
```

用一个最简化的式子表示，就是：

$$
a^* = \arg\min_a C\!\left(W(s,a)\right)
$$

其中，\(s\) 是当前状态，\(a\) 是候选动作，\(W(s,a)\) 是世界模型预测的后果，\(C\) 负责评价后果。机器不必把每种动作都在现实中试一遍，而是可以先在内部预演，再选择行动。

::: tip 预测为什么关键
没有这类预测，机器只能依靠已经学会的反应，或行动之后再看结果；有了预测，它才能在行动之前比较不同未来。**在这套架构中，规划就是利用预测来选择行动。**
:::

这里的“预测”不只意味着预测下一帧。按照这套架构，世界模型有两项任务：

1. 根据已经看到的内容，补全当前状态中缺失的空间或时间信息；
2. 根据自然变化或候选动作，预测一个或多个可能的未来状态。

因此，预测是世界模型服务于推理和规划的接口。但新的问题也随之出现：真实观测包含海量信息，世界模型究竟应该预测其中的什么？

### 预测观测，还是预测表示？

以驾驶为例。周围车辆会怎样运动，直接影响下一步操作；路边每片树叶会怎样摆动，通常与驾驶无关，也很难被精确确定。如果要求模型生成下一帧的全部像素，两类信息都会进入同一个预测任务。

这并不说明生成式预测没有价值。需要生成画面或进行精细仿真时，观测细节本来就是目标。JEPA 做的是另一种取舍：**不直接预测完整观测，而是预测由编码器提取的目标表示。**

| 预测目标     | 模型要回答什么                   | 主要取舍                                       |
| ------------ | -------------------------------- | ---------------------------------------------- |
| **完整观测** | 目标画面或信号具体是什么样       | 保留并生成细节，也要承担全部观测的信息量       |
| **抽象表示** | 从当前信息可以推断出哪些目标状态 | 可以省去不必重建的细节，但表示是否有用仍需检验 |

::: warning 不可预测，不等于不重要
另一辆车是否突然转向同样难以确定，却绝不能被忽略。可以抽象掉的是与任务无关的偶然细节；与行动有关的不确定性仍要被表达。LeCun 的原始蓝图为此引入潜变量 \(z\)，用来表示当前信息无法确定的多种可能。后来的具体 JEPA 实现并不都采用同一种做法。
:::

JEPA 由此出现：它仍然通过预测来学习，但把目标从观测本身移到了表示空间。它希望得到的不是一幅精确的未来图像，而是一种**有信息、可预测，并可能服务于后续规划的内部状态**。下一节将进一步拆开这个过程，说明 Joint Embedding Predictive Architecture 这个名字究竟对应哪些计算。

## 5.1.2　JEPA 的核心：在表示空间中预测

要看懂这句话，先要分清**原始信息**和**表示**。以遮挡图像为例：模型已经看到的区域是 context $x$，被遮住但真实存在的区域是 target $y$；Encoder 输出的内部向量或 token 序列，才是它们的表示 $s_x$ 和 $s_y$。表示不是另一张图，也不是人类预先写好的“位置”或“物体”等变量。

### 一次 JEPA 预测，数据怎样流动？

一次最简单的 JEPA 训练，可以沿着下面两条分支阅读。

<div class="jepa-flow-board" role="figure" aria-label="一次 JEPA 训练的数据流">
  <section class="jepa-flow-branch">
    <span class="jepa-flow-branch-label">预测分支</span>
    <div class="jepa-flow-chain">
      <div class="jepa-flow-node is-observation">
        <small>已知信息</small>
        <strong>context <code>x</code></strong>
        <span>可见区域</span>
      </div>
      <span class="jepa-flow-arrow" aria-hidden="true">→</span>
      <div class="jepa-flow-node">
        <small>编码</small>
        <strong>Context Encoder</strong>
      </div>
      <span class="jepa-flow-arrow" aria-hidden="true">→</span>
      <div class="jepa-flow-node is-state">
        <small>context 表示</small>
        <strong><code>s<sub>x</sub></code></strong>
      </div>
      <span class="jepa-flow-arrow" aria-hidden="true">→</span>
      <div class="jepa-flow-node is-predictor">
        <small>加入已知条件 <code>c</code></small>
        <strong>Predictor</strong>
      </div>
      <span class="jepa-flow-arrow" aria-hidden="true">→</span>
      <div class="jepa-flow-node is-prediction">
        <small>预测结果</small>
        <strong><code>ŝ<sub>y</sub></code></strong>
      </div>
    </div>
  </section>
  <section class="jepa-flow-branch">
    <span class="jepa-flow-branch-label">参照分支 · 真实 target 的表示仅在训练时参与损失</span>
    <div class="jepa-flow-chain">
      <div class="jepa-flow-node is-observation">
        <small>真实内容</small>
        <strong>target <code>y</code></strong>
        <span>被遮区域</span>
      </div>
      <span class="jepa-flow-arrow" aria-hidden="true">→</span>
      <div class="jepa-flow-node">
        <small>编码</small>
        <strong>Target Encoder</strong>
      </div>
      <span class="jepa-flow-arrow" aria-hidden="true">→</span>
      <div class="jepa-flow-node is-state">
        <small>target 表示</small>
        <strong><code>s<sub>y</sub></code></strong>
      </div>
    </div>
  </section>
  <div class="jepa-flow-compare">
    <code>ŝ<sub>y</sub></code>
    <strong>在表示空间中比较</strong>
    <code>s<sub>y</sub></code>
    <span>Predictor 不会看到真实 target 的内容。</span>
  </div>
</div>

_为突出核心关系，图中省略了实现细节：在 I-JEPA 中，Target Encoder 会先处理完整图像，再选出目标区域对应的特征。_

沿着这张图，一次训练只做四件事：

<div class="jepa-step-grid" role="list" aria-label="一次 JEPA 训练的四个步骤">
  <section class="jepa-step-card" role="listitem">
    <span>01</span>
    <strong>构造问题</strong>
    <p>把可见区域作为 context，把被遮区域作为 target；target 的位置是已知条件 <code>c</code>。</p>
  </section>
  <section class="jepa-step-card" role="listitem">
    <span>02</span>
    <strong>分别编码</strong>
    <p>两个 Encoder 分别得到 <code>s<sub>x</sub></code> 和 <code>s<sub>y</sub></code>。</p>
  </section>
  <section class="jepa-step-card" role="listitem">
    <span>03</span>
    <strong>预测表示</strong>
    <p>Predictor 根据 <code>s<sub>x</sub></code> 和 <code>c</code> 产生 <code>ŝ<sub>y</sub></code>。</p>
  </section>
  <section class="jepa-step-card" role="listitem">
    <span>04</span>
    <strong>提供训练信号</strong>
    <p>比较 <code>ŝ<sub>y</sub></code> 与 <code>s<sub>y</sub></code>，用二者的距离训练模型。</p>
  </section>
</div>

把这四步压缩成公式，就是：

$$
s_x = E_x(x), \qquad s_y = E_y(y)
$$

$$
\hat{s}_y = P(s_x,c), \qquad \mathcal{L}_{\mathrm{pred}} = D(\hat{s}_y,s_y)
$$

其中，$E_x$ 和 $E_y$ 是 Encoder，$P$ 是 Predictor，$D$ 是衡量两个表示差异的函数，$\mathcal{L}_{\mathrm{pred}}$ 是由此得到的预测损失。$c$ 是模型已经知道的条件：在这个例子中，它是 target 的位置；在其他任务中，也可以是预测的时间间隔或计划执行的动作。两个 Encoder 是否共享参数、具体怎样更新，不改变上面的基本数据流，这些训练细节将在 5.2 中展开。

完成这条最小流程后，再回看 JEA（Joint Embedding Architecture，联合嵌入架构）、生成式架构与 JEPA 的对比：

![JEA、生成式架构与 JEPA 的结构对比](/jepa/jepa-architecture-comparison.png)

![JEA、生成式架构与 JEPA 的纵向结构对比](/jepa/jepa-architecture-comparison-mobile.png)

_图 5.2　三类自监督学习架构的对比。JEA 直接比较两侧表示；生成式架构预测原始 target $y$；JEPA 预测 target 的表示 $s_y$。来源：[I-JEPA 论文 Figure 2](https://openaccess.thecvf.com/content/CVPR2023/papers/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.pdf)。_

这也解释了 JEPA 的名字：**Joint Embedding** 表示 context 与 target 都会被编码，**Predictive** 表示一侧的表示用于预测另一侧，**Architecture** 则说明它规定的是计算关系，而不是某种固定模态或网络。它与生成式预测最直接的区别是：Predictor 输出的是 target 表示，而不是 target 本身。

### 在表示空间预测，究竟改变了什么？

最关键的区别是：预测误差比较的是 $\hat{s}_y$ 与 $s_y$，不是 $\hat{y}$ 与 $y$。因此，原始 target 中的一项差异是否需要被预测，要依次回答两个问题。

<div class="jepa-decision" role="group" aria-label="target 中的差异怎样进入 JEPA 预测">
  <section class="jepa-decision-card is-ignored">
    <span>第一问：需要进入目标表示吗？</span>
    <strong>不需要：在表示中合并</strong>
    <p>如果不同的 target 被编码为相同或相近的 <code>s<sub>y</sub></code>，这项差异就不会进入预测误差，也不需要被恢复。</p>
  </section>
  <section class="jepa-decision-card is-kept">
    <span>如果需要保留，再问：能由 context 与条件确定吗？</span>
    <div class="jepa-decision-options">
      <div class="is-predictable">
        <small>能够确定</small>
        <strong>保留并预测</strong>
        <p>这项差异进入 <code>s<sub>y</sub></code>，Predictor 必须学会相应关系。</p>
      </div>
      <div class="is-uncertain">
        <small>无法唯一确定</small>
        <strong>需要表达不确定性</strong>
        <p>如果任务必须区分多个合理结果，单一确定性预测通常不够，还需要额外机制。</p>
      </div>
    </div>
  </section>
</div>

仍以车辆驶近岔路口为例：路面纹理可以不进入目标表示；车辆的位置和速度需要保留，并可能由已有信息推出；另一辆车最终左转还是右转同样重要，却未必能够提前确定。

::: info 潜变量 $z$ 是通用蓝图中的扩展
当一份 context 对应多个仍需区分的结果时，[LeCun 2022 年的通用 JEPA](https://openreview.net/forum?id=BZ5a1r-kVsf) 设想用潜变量 $z$ 参数化不同的兼容预测，以表达 context 中缺少的未知因素。这不是 JEPA 的必备模块；[I-JEPA](https://arxiv.org/abs/2301.08243) 和 [V-JEPA](https://arxiv.org/abs/2404.08471) 都没有显式使用 $z$ 来生成或搜索多个候选 target 表示。
:::

::: warning 预测误差小，不等于表示有用
JEPA 不会自动知道哪些差异应该保留。预测任务、数据和训练约束共同塑造 target 表示；如果所有输入都被编码成同一个常量，预测误差甚至也可以很小。怎样避免这种**表示坍缩**，将在 5.2 中展开。
:::

至此，JEPA 的基本计算关系已经清楚了：**构造 context 与 target，分别编码，用一侧表示预测另一侧表示，再用目标表示提供训练信号。** 后来的工作主要改变 context、target、预测条件以及预测结果的用途。下一节将沿着这些变化，梳理 JEPA 的发展过程。

## 5.1.3　JEPA 研究现状：一条主线与多条前沿

JEPA 的基本计算关系没有改变，变化的是**预测问题本身**：target 从静态图像中的区域扩展到视频中的时空状态，动作又进一步成为 Predictor 的条件。

下面先沿时间线观察这条能力主线怎样形成，再看当前研究正在分别补齐哪些瓶颈。前者回答“JEPA 怎样走到今天”，后者回答“它距离更完整的世界模型还缺什么”。

### 主线：从蓝图到受限规划

<div class="jepa-history" role="list" aria-label="JEPA 重要工作时间线">
  <article class="jepa-history-item" role="listitem">
    <div class="jepa-history-year">2022</div>
    <div class="jepa-history-dot" aria-hidden="true"></div>
    <div class="jepa-history-card">
      <div class="jepa-history-heading">
        <strong>提出核心设想：在抽象表示中预测世界</strong>
        <span class="jepa-history-tag is-blueprint">LeCun</span>
      </div>
      <p><b>核心变化：</b><a href="https://openreview.net/forum?id=BZ5a1r-kVsf">A Path Towards Autonomous Machine Intelligence</a> 提出，世界模型应预测抽象表示，而不是还原观测中的全部细节；H-JEPA 则进一步设想在多个抽象层级和时间尺度上预测。</p>
      <p class="jepa-history-boundary"><b>边界：</b>这是一份研究蓝图，不是已经完成的系统。</p>
    </div>
  </article>
  <article class="jepa-history-item" role="listitem">
    <div class="jepa-history-year">2023</div>
    <div class="jepa-history-dot" aria-hidden="true"></div>
    <div class="jepa-history-card">
      <div class="jepa-history-heading">
        <strong>在静态图像上验证表示预测</strong>
        <span class="jepa-history-tag is-representation">I-JEPA</span>
      </div>
      <p><b>预测关系：</b><code>可见区域 → 被遮挡区域的表示</code>。<a href="https://arxiv.org/abs/2301.08243">I-JEPA</a> 给出了可扩展的图像实验，说明这种目标可以学到有用的静态视觉表征。</p>
      <p class="jepa-history-boundary"><b>边界：</b>它不包含时间和动作，也不描述环境如何变化。</p>
    </div>
  </article>
  <article class="jepa-history-item" role="listitem">
    <div class="jepa-history-year">2024</div>
    <div class="jepa-history-dot" aria-hidden="true"></div>
    <div class="jepa-history-card">
      <div class="jepa-history-heading">
        <strong>从静态图像走向视频时空表征</strong>
        <span class="jepa-history-tag is-video">V-JEPA</span>
      </div>
      <p><b>预测关系：</b><code>可见时空区域 → 被遮挡时空区域的表示</code>。<a href="https://arxiv.org/abs/2404.08471">V-JEPA</a> 将遮挡表示预测扩展到视频，使模型从无动作标签的视频中学习兼顾外观与运动信息的时空表征。</p>
      <p class="jepa-history-boundary"><b>边界：</b>它不是严格的“过去预测未来”，也不建模动作造成的因果状态转移。</p>
    </div>
  </article>
  <article class="jepa-history-item" role="listitem">
    <div class="jepa-history-year">2025</div>
    <div class="jepa-history-dot" aria-hidden="true"></div>
    <div class="jepa-history-card">
      <div class="jepa-history-heading">
        <strong>让动作进入预测，并开始服务规划</strong>
        <span class="jepa-history-tag is-action">V-JEPA 2 → V-JEPA 2-AC</span>
      </div>
      <p><b>两步推进：</b><a href="https://arxiv.org/abs/2506.09985">V-JEPA 2</a> 先扩大视频预训练的模型与数据规模；随后，V-JEPA 2-AC 冻结视频 Encoder，用机器人轨迹训练动作条件 Predictor，并接入模型预测控制（MPC）。外部规划器（Planner，负责提出和选择候选动作）会比较动作的预测后果，再执行当前一步。</p>
      <p class="jepa-history-boundary"><b>边界：</b>基础 V-JEPA 2 不接收动作。在论文的机器人部署中，规划视野为 1；pick-and-place 还使用了两张人工给定的中间目标图像。这说明的是闭环、目标条件下的受限控制，而不是自主分解长程任务。</p>
    </div>
  </article>
</div>

::: tip 从一条主线到多个瓶颈
这条时间线说明，JEPA 已经从静态表征学习推进到能够服务于受限规划的动作条件预测。但“能够接入 Planner”距离“形成通用世界模型”仍然很远：状态要稳定学出，要保留行动需要的信息；Predictor 要描述动作和不确定未来；Planner 还要把短期预测组织成长程行动。

下面五条路线是五个**问题维度**，不是五类互斥模型。同一项工作可能同时涉及其中多条。
:::

### 前沿：五个瓶颈怎样相互衔接？

截至 2026 年 8 月，相关研究大致可以放进三个相互连接的层次：先让模型学得出来，再让预测对行动有用，最后把预测真正用于行动。这是理解问题的层次，不是必须依次完成的流水线。

<div class="jepa-frontier-map" role="list" aria-label="JEPA 前沿研究的三个层次">
  <section role="listitem">
    <span>训练层</span>
    <strong>先学得出来</strong>
    <p>防塌缩 · 训练架构</p>
  </section>
  <span class="jepa-frontier-map-arrow" aria-hidden="true">→</span>
  <section role="listitem">
    <span>模型层</span>
    <strong>再预测得有用</strong>
    <p>状态表示 · 动作动力学</p>
  </section>
  <span class="jepa-frontier-map-arrow" aria-hidden="true">→</span>
  <section role="listitem">
    <span>系统层</span>
    <strong>最后用于行动</strong>
    <p>Planner · 长程规划</p>
  </section>
</div>

<div class="jepa-frontier-grid" role="list" aria-label="2026 年 JEPA 的五条并行研究线">
  <article class="jepa-frontier-card is-training" role="listitem">
    <span class="jepa-frontier-index">01</span>
    <strong>防塌缩：怎样学出有信息的状态？</strong>
    <p>如果所有输入都被编码成同一个常量，预测误差也可能很小。I-JEPA 与 V-JEPA 结合停止梯度、EMA 更新的 Target Encoder 和不对称的 context/target 预测结构来稳定训练；<a href="https://proceedings.neurips.cc/paper_files/paper/2024/hash/04a80267ad46fc730011f8760f265054-Abstract-Conference.html">Contrastive-JEPA（防塌缩 C-JEPA）</a> 和 <a href="https://arxiv.org/abs/2511.08544">LeJEPA</a> 则进一步用显式正则约束表示的方差或分布。</p>
    <p class="jepa-frontier-note"><b>仍未解决：</b>不塌缩只是最低条件。表示彼此不同，不等于它保留了正确的语义、动力学和动作信息。</p>
  </article>
  <article class="jepa-frontier-card is-architecture" role="listitem">
    <span class="jepa-frontier-index">02</span>
    <strong>架构：Encoder 与 Predictor 应该怎样训练？</strong>
    <p>V-JEPA 2-AC 冻结大规模视频 Encoder，只训练动作条件 Predictor；<a href="https://arxiv.org/abs/2603.19312">LeWorldModel</a> 则让 Encoder 与 Predictor 端到端一起学习。前者复用已经学到的视觉表征，后者允许内部状态与环境动力学共同适配。</p>
    <p class="jepa-frontier-note"><b>仍未解决：</b>冻结路线可以直接复用大规模预训练表征，但表示空间未必为当前动力学任务设计；端到端路线允许预测目标反过来塑造表示，同时必须显式处理表示坍缩。两者在什么数据、任务和计算预算下更合适，目前缺少统一的直接比较。</p>
  </article>
  <article class="jepa-frontier-card is-interface" role="listitem">
    <span class="jepa-frontier-index">03</span>
    <strong>表示：怎样让内部状态真正适合行动？</strong>
    <p>精细控制需要保留空间位置、局部边界和时序一致性等信息，<a href="https://arxiv.org/abs/2603.14482">V-JEPA 2.1</a> 因此加强了稠密预测与深层监督。但一项 2026 年 TMLR 接收的<a href="https://openreview.net/forum?id=cHZn5Gdh8e">系统研究</a>也发现，在其评估的导航和操作设置中，较低的多步 rollout 误差并不稳定对应更高的规划成功率。</p>
    <p class="jepa-frontier-note"><b>仍未解决：</b>不塌缩、有信息、预测准确和适合规划，是不同层次的要求，不能相互替代。表示是否合适，仍需通过具体下游任务检验。</p>
  </article>
  <article class="jepa-frontier-card is-dynamics" role="listitem">
    <span class="jepa-frontier-index">04</span>
    <strong>动力学：怎样预测动作造成的未来？</strong>
    <p>视频只能告诉模型世界发生了变化，动作条件模型还要分辨变化由什么行为造成。一些工作用交互轨迹学习状态转移，另一些工作尝试从无动作标签视频中<a href="https://arxiv.org/abs/2601.05230">发现潜在动作</a>；也有工作开始训练和评估多步 rollout。</p>
    <p class="jepa-frontier-note"><b>仍未解决：</b>动作数据稀缺，多步预测会累积误差；如何进一步表示同一状态对应的多个合理未来，仍是尚未解决的问题。</p>
  </article>
  <article class="jepa-frontier-card is-planning" role="listitem">
    <span class="jepa-frontier-index">05</span>
    <strong>Planner：怎样把短期预测变成长程行动？</strong>
    <p>Predictor 只回答“执行这些动作可能发生什么”，Planner 才负责提出并选择动作。短期 MPC 搜索具体动作序列；<a href="https://arxiv.org/abs/2604.03208">HWM</a> 尝试先寻找较长时间尺度的潜在子目标，<a href="https://arxiv.org/abs/2606.09311">FF-JEPA</a> 则初步探索由模型直接预测子目标。</p>
    <p class="jepa-frontier-note"><b>仍未解决：</b>现有方法仍依赖给定目标、外部搜索或边界清晰的短任务。开放世界中的长期自主规划还没有实现。</p>
  </article>
</div>

### 横向探索：更多任务与预测方式

前面的五个瓶颈讨论的是怎样补齐 JEPA 从状态学习到行动规划的能力链。与此同时，还有一批工作通过改变模态、任务或 target 的定义，探索这套预测原则还能做什么：

- **改变预测层级：** [Causal-JEPA（对象级 C-JEPA）](https://arxiv.org/abs/2602.11389) 从局部区域进一步走向对象级状态及对象间相互作用。它与前文的 Contrastive-JEPA 只是缩写相同，并非同一项工作。
- **改变模态与任务：** [VL-JEPA](https://arxiv.org/abs/2512.10942) 将它扩展到视觉—语言任务；[A-JEPA](https://arxiv.org/abs/2311.15830) 与 [Point-JEPA](https://arxiv.org/abs/2404.16432) 则分别探索音频和点云。
- **统一不同数据形态：** [UniJEPA](https://arxiv.org/abs/2608.07409) 尝试在同一框架中建模图像与视频。

这些工作扩展的是 JEPA 的问题边界：context 与 target 可以怎样定义，模型可以预测什么，以及这种预测能够支持哪些任务。它们不是行动规划之后的“下一代 JEPA”。

_截至 2026 年 8 月，LeJEPA、LeWorldModel、V-JEPA 2.1、HWM、FF-JEPA 和 Temporal-Distance JEPA 等多项代表工作仍是预印本。它们展示的是正在形成的研究方向，而不是已经公认的最终架构。_

## 本节小结

JEPA 的核心不是某个固定模型，而是在表示空间建立预测关系：给定 context 及已知条件，预测 target 的内部表示，而不要求重建全部观测细节。

沿着一条能力主线，研究从静态图像区域预测扩展到视频时空表征，再通过动作条件 Predictor 与外部 Planner 进入受限控制；与此同时，也有工作在改变模态、任务和 target 的定义。所有工作都需要检验表示是否对目标任务有用、预测关系是否可靠；只有面向行动的世界模型，还需要进一步证明这些预测能够改善闭环规划与控制。

## 主要参考资料

- [LeCun, 2022, A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf)——JEPA、H-JEPA 与完整自主智能架构的原始蓝图。
- [I-JEPA](https://arxiv.org/abs/2301.08243)、[IWM](https://arxiv.org/abs/2403.00504)、[V-JEPA](https://arxiv.org/abs/2404.08471)、[V-JEPA 2](https://arxiv.org/abs/2506.09985) 与 [V-JEPA 2.1](https://arxiv.org/abs/2603.14482)——核对图像、条件预测、视频、动作条件预测和稠密局部表征的发展。
- [NeurIPS 2024 Contrastive-JEPA（防塌缩 C-JEPA）](https://proceedings.neurips.cc/paper_files/paper/2024/hash/04a80267ad46fc730011f8760f265054-Abstract-Conference.html)、[LeJEPA](https://arxiv.org/abs/2511.08544)、[LeWorldModel](https://arxiv.org/abs/2603.19312) 与 [When Does LeJEPA Learn a World Model?](https://arxiv.org/abs/2605.26379)——核对显式防塌缩、端到端世界模型及其理论边界。
- [Learning Latent Action World Models in the Wild](https://arxiv.org/abs/2601.05230)、[What Drives Success in Physical Planning with JEPA World Models?](https://openreview.net/forum?id=cHZn5Gdh8e)、[Temporal-Distance JEPA](https://arxiv.org/abs/2607.25337)、[HWM](https://arxiv.org/abs/2604.03208) 与 [FF-JEPA](https://arxiv.org/abs/2606.09311)——核对潜在动作、规划接口与长程规划的早期探索。
- [Causal-JEPA（对象级 C-JEPA）](https://arxiv.org/abs/2602.11389)、[VL-JEPA](https://arxiv.org/abs/2512.10942)、[A-JEPA](https://arxiv.org/abs/2311.15830)、[Point-JEPA](https://arxiv.org/abs/2404.16432) 与 [UniJEPA](https://arxiv.org/abs/2608.07409)——核对对象状态及相互作用、视觉语言、音频、点云和统一建模等横向扩展。
- [Stanford CS25：Joint Embedding Predictive World Models](https://web.stanford.edu/class/cs25/)（[slides](https://drive.google.com/file/d/1bF5Yfzf-FG5iNIAgsXn2DwVD3l3ymvZW/view)）——采用“状态表征—世界变化—动作后果”的教学骨架。
- [LeCun, Les Houches 2022, Lecture 3](https://leshouches2022.github.io/SLIDES/lecun-20220720-leshouches-03.pdf)——核对 JEPA、latent variable、H-JEPA 的原始概念关系。
- [LeCun & Manyika, 2026, Learning Abstractions](https://www.amacad.org/publication/daedalus/learning-abstractions-conversation-yann-lecun)——核对“预测充分的抽象表示”与多层、长时间尺度预测的近期表述。
- [LeCun, NUS 2025](https://drive.google.com/file/d/1j3A8SYtkJqui7CotCMuvES2LpR17JYrQ/view)——参考“寻找可预测状态变量”到动作条件世界模型的宏观过渡。
- [LeCun, Harvard 2024](https://cmsa.fas.harvard.edu/media/lecun-20240328-harvard_reduced.pdf)——核对 JEPA、世界模型与完整 Objective-Driven AI 架构的边界。
- [I-JEPA, CVPR 2023 slides](https://cvpr.thecvf.com/media/cvpr-2023/Slides/21019.pdf)——核对 I-JEPA 的最小训练结构与实验设置。
- [NYU Embodied Learning and Vision：World Models and Planning](https://elvcourse.org/course-public/2026-spring/lectures/week05_wm_planning.pdf)——用独立课程视角校正 JEPA 在整个世界模型谱系中的位置。

[回到第 5 章](./index.md)
