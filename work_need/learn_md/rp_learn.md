
风险平价(Risk-Parity)

参考资料:

[你真的搞懂了风险平价吗？ - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/38301218)

[资产瞎配模型（三）：风险平价及其优化 - 腾讯云开发者社区-腾讯云 (tencent.com)](https://cloud.tencent.com/developer/article/1397794)

理论基础:

    模型来自于上世纪90年代桥水的Ray.Dalio在投资中对于收益率的分析建模，广义上的经济增长，有关键因子：经济增长和通胀，这是经济学里面驱动经济机器运行的一种观点。

    那根据这两个因子不同，我们进行组合得到下列组合[2X2]

    [“经济上升",“通胀上升”],

    [“经济上升",“通胀下降”],

    [“经济下降”,“通胀上升"],

    [“经济下降”,“通胀下降"]

    假设基本可以描述一个基本的经济周期。

    当我们知道了在每种经济环境中应该投资哪种投资品之后，最关键问题就是：未来一段时间属于什么经济环境,该如何配置资产？

    **桥水给出的答案是：“不知道”也“不猜”,不尝试去预测！**

    全天候理论核心在于，是被动的，保证自己的资产组合在四种经济环境中有着相同的风险暴露，从而对冲市场环境的风险，

    使得未来无论**处于哪一种经济环境**，**该投资组合的风险都是可控的**。

公式推导部分:

基本模型

给定资产价格

$	rt = \frac{p_t}{p_{t-1}} - 1$

使用对数收益率也可以

$u = E[r_t] = \frac{1}{T}∑\limits_{t=1}^{T}r_t$

资产的波动率为

$σ =\sqrt{Var(r_t)}$

两种资产的协方差

$σ_{1,2}= cov(r_1,r_2)= \rho_{12}\sigma_1\sigma_2$

假设共有p种资产可以配置的资产，预期回报、波动率、协方差分别为

$1)r= [r_1,r_2,...,r_n]^T$

$2)\sigma = [\sigma_1,\sigma_2,...,\sigma_n]^T$

$3)Σ = (\sigma_{ij})_{n*n}$

对于每类资产，配置权重部分

 $ w = [w_1,w_2,...,w_n]^T$

则组合P的预期回报率，波动率为

$$
\begin{cases}
r_p = ∑\limits_{i=1}^{n}w_ir_i =w^Tr \\ \sigma^2_p = Var(r_p) = ∑\limits_{i=1}^{n}∑\limits_{j=1}^{n}w_iw_j\sigma_{ij} = w^TΣw
	\end{cases}
$$

由矩阵求导法则

    $\frac{ \partial (w^TΣw) }{ \partial w } = (Σ^T + Σ)w = 2Σw$

对于组合，风险就是

  $  \sigma = \sqrt{w^TΣw}$

任一单个资产i的边际风险贡献(Marginal Risk Contribution)的定义就是:

$MRC_i = \frac{ \partial \sigma }{ \partial w_i } = \frac{(Σw)}{\sqrt{w^TΣw}}$

那么，资产对总风险的贡献为该资产权重与其边际风险贡献的乘积

$TRC_i = w_i\frac{ \partial \sigma_p }{ \partial w_i } = w_i\frac{cov(r_i,r_p)}{\sigma_p}$

来，我们来回顾一下风险平价的定义，**保证所有资产组合的风险暴露相同**

$∵ TRC = \sqrt{(w^TΣw)}$

由模型假设,各大资产的风险贡献相同,不妨把总风险平摊到每个资产类型上

    $BRC_i = TRC * [1/ N,...,1/N]$

其中,N为配置资产类型的数量

让后按照求解下列式

   $\mathop{min}\limits_{w} sum(TRC_i-BRC_i)^2 \\ s.t. \sum\limits_{i=1}^{n} w_i = 1,0<w_i<1$

得到权重

3)风险平价 --针对半衰期调整的算法

协方差衰减计算公式(针对时序公式)推导

其中,

    参数α为确认半衰期的衰减因子

模型中半衰期因子为

  $\lambda = \frac{1}{2}^{(1/T)}$，T为设置的半衰期的天数

pandas中ewm的计算公式:

   $		 y_t = \sum\limits_{i=0}^{i=n}\frac{(1-a)^ix_{t-i}}{(1-a)^i}$,

    其中,$\alpha为(1-\lambda)$

调整后的协方差计算:

    $\lambda_{i*1} = [\lambda^{n},...,\lambda^{0}]$

    $fraw = [(Ret_i - RetEwma_i) * (Ret_j - RetEwma_j)]*\lambda_{i*1} / sum(\lambda_{i*1})$


在fraw基础上进行Newey-West调整

    D为延迟算子

    $C_{plus} = \lambda_{[0,n-d]*1} * [(Ret_{i[d,n]} - RetEwma_i) * (Ret_j - RetEwma_j)]/sum(\lambda_{[0,n-d]*1})$

    $C_{minus} = \lambda_{[0,n-d]*1} * [(Ret_i - RetEwma_i) * (Ret_j - RetEwma_j)]/sum(\lambda_{[0,n-d]*1})$
