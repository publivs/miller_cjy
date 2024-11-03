# 用quantlib来计算期权
import QuantLib as ql
import numpy as np
import pandas as pd

# 配置日期计算条款
calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
dayCounter = ql.Actual365Fixed(ql.Actual365Fixed.Standard)

todayDate = ql.Date(11, ql.July, 2019)
maturity = todayDate + ql.Period(20, ql.Weeks)
settlementDate = todayDate

# 配置期权参数
stock = 49.0
strike = 50.0
riskFreeRate = 0.05
volatility = 0.2

# 配置全局估值日期
ql.Settings.instance().evaluationDate = todayDate
dayCounter = ql.Actual365Fixed(ql.Actual365Fixed.Standard)

todayDate = ql.Date(11, ql.July, 2019)
maturity = todayDate + ql.Period(20, ql.Weeks)
settlementDate = todayDate

# 配置期权参数
stock = 49
strike = 50
riskFreeRate = 0.05
volatility = 0.2

# 配置全局估值日期
ql.Settings.instance().evaluationDate = todayDate


# 配置行权条款
europeanExercise = ql.EuropeanExercise(maturity)
optionType = ql.Option.Call
payoff = ql.PlainVanillaPayoff(
    type=optionType, strike=strike)

# 构建期权对象
europeanOption = ql.VanillaOption(
    payoff=payoff,
    exercise=europeanExercise)

underlying = ql.SimpleQuote(stock)
underlyingH = ql.QuoteHandle(underlying)

# 无风险利率的期限结构
flatRiskFreeTS = ql.YieldTermStructureHandle(
    ql.FlatForward(
        settlementDate, riskFreeRate, dayCounter))

# 波动率的期限结构
flatVolTS = ql.BlackVolTermStructureHandle(
    ql.BlackConstantVol(
        settlementDate, calendar,
        volatility, dayCounter))

# 构建 BS 过程
bsProcess = ql.BlackScholesProcess(
    s0=underlyingH,
    riskFreeTS=flatRiskFreeTS,
    volTS=flatVolTS)

# 基于 BS 过程的公式定价引擎
pricingEngine = ql.AnalyticEuropeanEngine(
    bsProcess)

europeanOption.setPricingEngine(pricingEngine)

# RESULTS
print("Option value =", europeanOption.NPV())
print("Delta value  =", europeanOption.delta())
print("Theta value  =", europeanOption.theta())
print("Theta perday =", europeanOption.thetaPerDay())
print("Gamma value  =", europeanOption.gamma())
print("Vega value   =", europeanOption.vega())
print("Rho value    =", europeanOption.rho())

# europeanOption.NPV()
# europeanOption.delta()
# europeanOption.theta()
# europeanOption.thetaPerDay()
# europeanOption.gamma()
# europeanOption.vega()
# europeanOption.rho()