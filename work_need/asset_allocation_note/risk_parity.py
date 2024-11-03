import pandas as pd 
import numpy as np 

def half_life_method(Pa, est_ret, d_date):
    # half_life = 60    # 半衰期 表示t-60天的权重为t天的1/2
    if d_date.month == 12:
        half_life = Pa.half_life_weight_saa
    else:
        half_life = Pa.half_life_weight_taa

    lamda = 0.5 ** (1 / half_life)  # 这里的意图是什么,模型基础是啥
    h = len(est_ret) - 1  # 样本数量-1
    lamda_list = list(lamda ** (np.array(range(h + 1))))
    lamda_list = lamda_list[::-1]

    ewma_h = est_ret.ewm(alpha=1 - lamda).mean()

    covmat = np.zeros((len(est_ret.columns), len(est_ret.columns)))
    for i, col1 in enumerate(est_ret.columns):
        for j, col2 in enumerate(est_ret.columns):
            fk = est_ret[col1].values
            fl = est_ret[col2].values
            fk_ewma = ewma_h.loc[est_ret.index[-1], col1]
            fl_ewma = ewma_h.loc[est_ret.index[-1], col2]

            # 加权平均
            Fraw = np.dot(lamda_list, np.multiply(fk - fk_ewma, fl - fl_ewma)) / sum(lamda_list)
            # 在𝐹𝑅𝑎𝑤的基础上进行 Newey-West 调整
            D = 2  # 表示滞后时间长度
            Fnw = 0
            # 这里在干嘛,为啥就截了三个
            for k in range(1, D + 1):
                C_plus = np.dot(lamda_list[k:], np.multiply(fk[:-k] - fk_ewma, fl[k:] - fl_ewma) ) / sum(lamda_list[k:])

                C_minus = np.dot(lamda_list[k:], np.multiply(fk[k:] - fk_ewma, fl[:-k] - fl_ewma)) / sum(lamda_list[k:])

                Fnw += (1 - k / (D + 1)) * (C_plus + C_minus)

            Fnw = Fraw + Fnw
            covmat[i, j] = Fnw
    return covmat