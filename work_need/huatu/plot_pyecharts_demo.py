
import pandas as pd 
import json
import numpy as np 

import pyecharts.charts as pchart
from pyecharts.charts import Bar,Line
import pyecharts.options as opts
from pyecharts.globals import ThemeType

multi_bar_dict_volume = '''{
    "traceId": "1593057449793865408",
    "code": 200,
    "message": null,
    "size": 3,
    "headers": [
        {
            "columnSeqNo": 0,
            "columnName": "tradeDate",
            "columnCnName": "日期",
            "dataType": "DATE",
            "unitType": null,
            "primary": null
        },
        {
            "columnSeqNo": 1,
            "columnName": "name",
            "columnCnName": "指标名称",
            "dataType": "STRING",
            "unitType": null,
            "primary": null
        },
        {
            "columnSeqNo": 2,
            "columnName": "currentAmount",
            "columnCnName": "本周日均成交额",
            "dataType": "DOUBLE",
            "unitType": null,
            "primary": null
        },
        {
            "columnSeqNo": 3,
            "columnName": "lastWeekAmount",
            "columnCnName": "前1周日均成交额",
            "dataType": "DOUBLE",
            "unitType": null,
            "primary": null
        },
        {
            "columnSeqNo": 4,
            "columnName": "twoWeekAmount",
            "columnCnName": "前2周日均成交额",
            "dataType": "DOUBLE",
            "unitType": null,
            "primary": null
        },
        {
            "columnSeqNo": 5,
            "columnName": "threeWeekAmount",
            "columnCnName": "前3周日均成交额",
            "dataType": "DOUBLE",
            "unitType": null,
            "primary": null
        },
        {
            "columnSeqNo": 6,
            "columnName": "fourWeekAmount",
            "columnCnName": "前4周日均成交额",
            "dataType": "DOUBLE",
            "unitType": null,
            "primary": null
        },
        {
            "columnSeqNo": 7,
            "columnName": "fiveWeekAmount",
            "columnCnName": "前5周日均成交额",
            "dataType": "DOUBLE",
            "unitType": null,
            "primary": null
        },
        {
            "columnSeqNo": 8,
            "columnName": "sixWeekAmount",
            "columnCnName": "前6周日均成交额",
            "dataType": "DOUBLE",
            "unitType": null,
            "primary": null
        }
    ],
    "body": [
        {
            "#3": 433489803103.5,
            "#4": 476261146266.8,
            "#5": 491952923488.6,
            "#6": 556324240458.9,
            "#7": 493565554762.2,
            "#8": 434625236058.4,
            "#0": 20220107,
            "#1": "上证指数",
            "#2": 502891435426.3
        },
        {
            "#3": 245454412418.81,
            "#4": 247756248294.36,
            "#5": 232214881751.47,
            "#6": 257239395158.48,
            "#7": 237579924048.55,
            "#8": 243613185732.58,
            "#0": 20220107,
            "#1": "中小综指",
            "#2": 263703191217.05
        },
        {
            "#3": 239304169523.54,
            "#4": 253485491513.42,
            "#5": 288897883312.41,
            "#6": 263698539318.69,
            "#7": 256855960900,
            "#8": 303119070980.38,
            "#0": 20220107,
            "#1": "创业板综",
            "#2": 264976808820.33
        }
    ],
    "aggregation": null
}
'''

multi_bar_ratio = '''{
    "traceId": "1593057449793865408",
    "code": 200,
    "message": null,
    "size": 3,
    "headers": [
        {
            "columnSeqNo": 0,
            "columnName": "tradeDate",
            "columnCnName": "日期",
            "dataType": "DATE",
            "unitType": null,
            "primary": null
        },
        {
            "columnSeqNo": 1,
            "columnName": "name",
            "columnCnName": "指标名称",
            "dataType": "STRING",
            "unitType": null,
            "primary": null
        },
        {
            "columnSeqNo": 2,
            "columnName": "currentAmount",
            "columnCnName": "本周日均成交额",
            "dataType": "DOUBLE",
            "unitType": null,
            "primary": null
        },
        {
            "columnSeqNo": 3,
            "columnName": "lastWeekAmount",
            "columnCnName": "前1周日均成交额",
            "dataType": "DOUBLE",
            "unitType": null,
            "primary": null
        },
        {
            "columnSeqNo": 4,
            "columnName": "twoWeekAmount",
            "columnCnName": "前2周日均成交额",
            "dataType": "DOUBLE",
            "unitType": null,
            "primary": null
        },
        {
            "columnSeqNo": 5,
            "columnName": "threeWeekAmount",
            "columnCnName": "前3周日均成交额",
            "dataType": "DOUBLE",
            "unitType": null,
            "primary": null
        },
        {
            "columnSeqNo": 6,
            "columnName": "fourWeekAmount",
            "columnCnName": "前4周日均成交额",
            "dataType": "DOUBLE",
            "unitType": null,
            "primary": null
        },
        {
            "columnSeqNo": 7,
            "columnName": "fiveWeekAmount",
            "columnCnName": "前5周日均成交额",
            "dataType": "DOUBLE",
            "unitType": null,
            "primary": null
        },
        {
            "columnSeqNo": 8,
            "columnName": "sixWeekAmount",
            "columnCnName": "前6周日均成交额",
            "dataType": "DOUBLE",
            "unitType": null,
            "primary": null
        }
    ],
    "body": [
        {
            "#3": 433489803103.5,
            "#4": 476261146266.8,
            "#5": 491952923488.6,
            "#6": 556324240458.9,
            "#7": 493565554762.2,
            "#8": 434625236058.4,
            "#0": 20220107,
            "#1": "上证指数",
            "#2": 502891435426.3
        },
        {
            "#3": 245454412418.81,
            "#4": 247756248294.36,
            "#5": 232214881751.47,
            "#6": 257239395158.48,
            "#7": 237579924048.55,
            "#8": 243613185732.58,
            "#0": 20220107,
            "#1": "中小综指",
            "#2": 263703191217.05
        },
        {
            "#3": 239304169523.54,
            "#4": 253485491513.42,
            "#5": 288897883312.41,
            "#6": 263698539318.69,
            "#7": 256855960900,
            "#8": 303119070980.38,
            "#0": 20220107,
            "#1": "创业板综",
            "#2": 264976808820.33
        }
    ],
    "aggregation": null
}
'''

fund_financial_operation = '''
{
    "traceId": "1593057449802253992",
    "code": 200,
    "message": null,
    "size": 24,
    "headers": [
        {
            "columnSeqNo": 0,
            "columnName": "tradeDate",
            "columnCnName": "日期",
            "dataType": "DATE",
            "unitType": null,
            "primary": null
        },
        {
            "columnSeqNo": 1,
            "columnName": "20010101",
            "columnCnName": "股票型基金",
            "dataType": "DOUBLE",
            "unitType": null,
            "primary": null
        },
        {
            "columnSeqNo": 2,
            "columnName": "20010102",
            "columnCnName": "混合型基金",
            "dataType": "DOUBLE",
            "unitType": null,
            "primary": null
        }
    ],
    "body": [
        {
            "#0": 20201231,
            "#1": null,
            "#2": 0.01527375
        },
        {
            "#0": 20210101,
            "#1": null,
            "#2": null
        },
        {
            "#0": 20210104,
            "#1": 0.02514527,
            "#2": 0.09405223
        },
        {
            "#0": 20210105,
            "#1": null,
            "#2": 0.01961781
        },
        {
            "#0": 20210106,
            "#1": 0.0007019,
            "#2": 0.06235009
        },
        {
            "#0": 20210107,
            "#1": null,
            "#2": 0.03124601
        },
        {
            "#0": 20210108,
            "#1": null,
            "#2": null
        },
        {
            "#0": 20210111,
            "#1": 0.02480676,
            "#2": 0.08902496
        },
        {
            "#0": 20210112,
            "#1": null,
            "#2": 0.00872171
        },
        {
            "#0": 20210113,
            "#1": 0.02231899,
            "#2": 0.01686759
        },
        {
            "#0": 20210114,
            "#1": 0.0020388,
            "#2": 0.02169751
        },
        {
            "#0": 20210115,
            "#1": null,
            "#2": 0.01827319
        },
        {
            "#0": 20210118,
            "#1": 0.01052866,
            "#2": 0.11367386
        },
        {
            "#0": 20210119,
            "#1": 0.00248275,
            "#2": 0.00961619
        },
        {
            "#0": 20210120,
            "#1": 0.00190229,
            "#2": 0.05298372
        },
        {
            "#0": 20210121,
            "#1": 0.00035333,
            "#2": 0.008
        },
        {
            "#0": 20210122,
            "#1": null,
            "#2": 0.03904327
        },
        {
            "#0": 20210123,
            "#1": 0.00041588,
            "#2": null
        },
        {
            "#0": 20210125,
            "#1": 0.02910827,
            "#2": 0.05610439
        },
        {
            "#0": 20210126,
            "#1": 0.00089269,
            "#2": 0.01735505
        },
        {
            "#0": 20210127,
            "#1": 0.00389885,
            "#2": 0.03900397
        },
        {
            "#0": 20210128,
            "#1": 0.01807152,
            "#2": 0.03533258
        },
        {
            "#0": 20210129,
            "#1": 0.00087765,
            "#2": null
        },
        {
            "#0": 20210130,
            "#1": null,
            "#2": 0.00737058
        }
    ],
    "aggregation": null
}
                            '''


# 成交额
multi_bar_dict_volume = json.loads(multi_bar_dict_volume)
value_df = pd.DataFrame(multi_bar_dict_volume['body'])
columns_cn_name = [dict_i['columnCnName'] for dict_i in multi_bar_dict_volume['headers']]
columns_name = [dict_i['columnName'] for dict_i in multi_bar_dict_volume['headers']]
value_df = value_df[value_df.columns.sort_values()]
value_df.columns = columns_name
value_df = value_df.drop(columns='tradeDate')
value_df = value_df.set_index('name')

value_df = value_df.astype('int64')
for i in value_df.index:
    for j in value_df.columns:
        print(i,j)
        value_df.loc[i,j]  = int(value_df.loc[i,j])
x_data = value_df.index.unique()

bar = (Bar()
            .add_xaxis(x_data.to_list())
            .add_yaxis('本周',list(value_df['currentAmount'].values.tolist()))
            .add_yaxis('第一周',list(value_df['lastWeekAmount'].values.tolist()))
            .add_yaxis('第二周',list(value_df['twoWeekAmount'].values.tolist()))
            .add_yaxis('第三周',list(value_df['threeWeekAmount'].values.tolist()))
            .add_yaxis('第四周',list(value_df['fourWeekAmount'].values.tolist()))
            .add_yaxis('第五周',list(value_df['fiveWeekAmount'].values.tolist()))
            .add_yaxis('第六周',list(value_df['sixWeekAmount'].values.tolist()))
            .set_series_opts(label_opts=opts.LabelOpts(font_size=8)) # 不显示数字
            .set_global_opts(
                            legend_opts=opts.LegendOpts(selected_mode=False,pos_bottom=True),
                            yaxis_opts=opts.AxisOpts(
                                                    splitline_opts = opts.SplitLineOpts(is_show=True)
                                            # axistick_opts=opts.AxisTickOpts(is_inside=True) ,#方向朝内,
                                                    )
                            )
            )
bar.render_notebook()




# 全球股指
global_index_value = (np.random.randint(0,10,8) -3)/100
index_col = [
            'RUS',
            '深圳成指',
            '上证综指',
            '巴西INDEX',
            'SP500',
            '富时100',
            '恒生指数',
            '德国DAX'
            ]
index_df = pd.DataFrame([global_index_value],columns=index_col)
bar=(
    Bar()
    .add_xaxis(index_df.columns.to_list())
    .reversal_axis()
    # .set_series_opts(label_opts=opts.LabelOpts(position="right"))
    .set_global_opts(title_opts=opts.TitleOpts(title="Index_float_rev"),
                    xaxis_opts=opts.AxisOpts(splitline_opts = opts.SplitLineOpts(is_show=True)
                                            # axistick_opts=opts.AxisTickOpts(is_inside=True) ,#方向朝内,
                                                    ))
    )
bar.add_yaxis('涨跌福',index_df.values.tolist()[0],bar_width=15,color='b')
bar.render_notebook()

# 获取不同的line
fund_financial_operation = json.loads(fund_financial_operation)

# column_info
column_name  = [ dict_i['columnCnName'] for dict_i in fund_financial_operation['headers'] ]
value_df = pd.DataFrame(fund_financial_operation['body'])
value_df.columns = column_name
value_df['日期'] = value_df['日期'].apply(lambda x:pd.to_datetime(str(x)))




# 基金发型走势图
x=value_df['日期'].to_list()
y1= value_df['股票型基金'].ffill().bfill().values.tolist()
y2= value_df['混合型基金'].ffill().bfill().values.tolist()
line=(
    Line()
    .add_xaxis(xaxis_data=x)
    .add_yaxis(series_name="股票型基金",y_axis=y1,symbol="line",is_symbol_show=False,
                linestyle_opts=opts.LineStyleOpts(color="orange", width=1,)
                )
    .add_yaxis(series_name="混合型基金",y_axis=y2,symbol="line",is_symbol_show=False,
                linestyle_opts=opts.LineStyleOpts(color="blue", width=1,))
    .set_global_opts(
                    title_opts=opts.TitleOpts(title="Line-多折线重叠"),
                    yaxis_opts=opts.AxisOpts(
                                            type_="value",
                                            axistick_opts=opts.AxisTickOpts(is_show=True),
                                            splitline_opts=opts.SplitLineOpts(is_show=True),
                                            ),
                    legend_opts=opts.LegendOpts(selected_mode=False,pos_bottom=True),
                    )
)
line.render_notebook()



#  "codes":"W00003.NYSE.IDX,W00001.NYSE.IDX,W00002.NYSE.IDX,W00032.OT.IDX,W00030.OT.IDX,W00031.OT.IDX,W00004.TSE.IDX",
#   	"globalParameters": {
# 		"where": ""
# 	},
# 	"functionParameters": [{
# 		"service": "StockMarket",
#         "tradeDate":19901231
# 	}],
# 	"pageNum": 1,
# 	"pageSize": "100"
# }