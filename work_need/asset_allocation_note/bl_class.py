class Bl_parameter:

    def __init__(self, req_dic):
        self.req_dic = req_dic
        #1.经常性调仓
        self._start_date = None # 回测区间起始日期, 如月度调仓则最好为月初
        self._end_date = None  # 回测区间截止日期, 如月度调仓则最好为月末
        self.measure = req_dic['measure']  # 选择风险衡量指标, 可选 'std' / 'semi-dev' / 'VaR' / 'CVaR'/ 'MDD'
        self.target = req_dic['target']  # 优化目标, target 可选 'optimize' / 'constraint',即 无条件优化/有条件优化
        self.obj = req_dic["obj"]   # 选择 'target' 后对应的优化目标, 如选择 optimize, 则默认参数 obj 只可选 'max_sharpe' / 'min_variance'
                                    # 如选择 'constraint', 则默认参数 obj 只可选 'target_return' / 'target_risk',
                                    # 并需要输入 target_return / target_risk
        self._target_return = None  # 目标回报   当优化目标为'有条件优化'且目标为'target_return'时，为必传参数,其他情况可不传
        self._target_risk = None # 目标风险  当优化目标为'有条件优化'且目标为'target_risk'时,为必传参数,其他情况可不传
        self._leverage = None   # 资产杠杆率, 如不指定则无杠杆, e.g. leverage = {'中证全债': 1.2} 为中证全债 1.2倍杠杆
        self.asset_name = req_dic["asset_name"]  # ['国债', '金融债', '信用债', '可转债', '长期存款', '股票', '股基', '债基', '货币']
        self.asset_list = req_dic["asset_list"]   #资产代码 ['0001a','0002a','0003a']
        self.rebalance_freq = req_dic["rebalance_freq"]  # 投资组合调仓频率, 可选 daily / weekly/ monthly, 默认为 monthly
        self.rebalance_period = req_dic["rebalance_period"]  # 投资组合调仓时间长度, 与 rebalance_freq 结合使用,
                                                # e.g. rebalance_freq = 'monthly', rebalance_period = 3, 即为每隔3个月调仓
        self.estimate_freq = req_dic["estimate_freq"]  # 投资组合参数估计频率, 可选 daily / weekly/ monthly, 默认为 monthly
        self.estimate_period = req_dic["estimate_period"]  # 投资组合参数估计时间长度, 与 estimate_freq 结合使用,
                                    # e.g. estimate_freq = 'monthly', estimate_period = 6, 即为通过调仓日过去6个月资产回报估计参数
        self._weq = None  # 资产均衡权重
        self._P = None  # 观点矩阵
        self._Q = None  # 观点回报收益率
        self._confidence = None  # 观点置信度
        self._bnds = None  # 单资产约束
        self._cons = None  # 多资产约束
        self._nav_data_df = None  # 指数数据
        self._ytd_asset_codes = None  # 使用到期收益率的资产编码

        # 非必传参数
        self._asset_fix_weight = None  # 资产固定的权重
        self._asset_fix_return = None  # 固定权重的资产的收益率
        self._method = None  # 如选择 VaR / CVaR作为风险衡量指标, 则此项为其估计方法,
                                # 可选 para / non-para (参数法/非参数法), 默认为非参数法
        self.ret_estmethod = req_dic["ret_estmethod"]  # 收益率估算方法, 可选 hist / exp (历史估计法 / 自定义)
        self.std_estmethod = req_dic["std_estmethod"]  # 波动率估算方法, 可选 hist / exp (历史估计法 / 自定义)
        self.exp_return = req_dic["exp_return"]
        self.exp_risk = None

        self._prob = None  # 如选择 VaR / CVaR作为风险衡量指标, 则此项为其置信度, 默认为 0.95
        self.data_freq = req_dic["data_freq"]  # 资产数据频率, 可选 daily / weekly/ monthly, 默认为 daily
        self.lever_cost = req_dic["lever_cost"]  # 杠杆成本计算使用的利率, 默认为 'SHIBOR3月'

        # 写死的参数
        self.scale = {'daily': 250, 'weekly': 52, 'monthly': 12}  # 各频率数据对应的交易日天数
        self.rf_list = {2015: 2.90, 2016: 2.45, 2017: 3.2, 2018: 3.5, 2019: 3.4}  # 自定义每年无风险利率,
                         # 用于计算 Sharpe_ratio, 如不指定则默认为0, 当前数据为银行间质押7日每年平均 (WIND代码:DR007)
        self.rf_allyears = 3.376  # mean of 10_year_bond
        self.rf = 0  # 有效前沿计算无风险利率, 默认为 0
        self.delta = 1  # 风险厌恶系数, 默认为 2.5

        #由传参得到的变量, 在此处先处理好，方便模型使用
        self._ret = None  # 收益率序列
        self._lever = None  # 杠杆比例
        self._risk_free = None  # 杠杆成本计算利率, 默认为 SHIBOR 3月
        self.asset_name_code_map = dict(zip(self.asset_name, self.asset_list))

