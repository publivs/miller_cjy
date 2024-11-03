import pandas as pd
import numpy as np
import time

node = {"children": [
			{
				"children": [
					{
						"children": [
							{
								"dicJsonDtoList": [
									{
										"dictionaryCodeName": "50502 股票——财富管理",
										"dictionaryCode": "505020101"
									},
									{
										"dictionaryCodeName": "50501 平衡型——财富管理",
										"dictionaryCode": "505010101"
									}
								],
								"tagCodeDet": "DETAIL_ASSET_CODE_NEW_LABEL",
								"children": [],
								"opSignDet": "in",
								"rel": " ",
								"dictionaryCode": [
									"505020101",
									"505010101"
								]
							},
							{
								"dicJsonDtoList": [
									{
										"dictionaryCodeName": "CNY",
										"dictionaryCode": "1"
									}
								],
								"tagCodeDet": "CURRENCY_CD_NEW_LABEL",
								"opValueDetText": "",
								"children": [],
								"opSignDet": "in",
								"rel": "and",
								"dictionaryCode": [
									"1"
								]
							},
							{
								"prefix": " case when",
								"suffix": " then false else true end ",
								"children": [
									{
										"children": [
											{
												"dicJsonDtoList": [
													{
														"dictionaryCodeName": "601398.SH-工商银行",
														"dictionaryCode": "601398.SH"
													},
													{
														"dictionaryCodeName": "601229.SH-上海银行",
														"dictionaryCode": "601229.SH"
													},
													{
														"dictionaryCodeName": "A02105.PF-大盘价值资产管理计划",
														"dictionaryCode": "A02105.PF"
													},
													{
														"dictionaryCodeName": "A02160.PF-广发特定策略1号特定多客户资产管理计划",
														"dictionaryCode": "A02160.PF"
													},
													{
														"dictionaryCodeName": "A02129.PF-泓德金融蓝筹1号资产管理计划",
														"dictionaryCode": "A02129.PF"
													},
													{
														"dictionaryCodeName": "A02132.PF-泓德金融蓝筹2号资产管理计划",
														"dictionaryCode": "A02132.PF"
													},
													{
														"dictionaryCodeName": "A02133.PF-泓德金融蓝筹3号资产管理计划",
														"dictionaryCode": "A02133.PF"
													},
													{
														"dictionaryCodeName": "A02176.PF-汇添富-添富牛53号资产管理计划",
														"dictionaryCode": "A02176.PF"
													}
												],
												"tagCodeDet": "TAA_SECURITY_ID",
												"opValueDetText": "",
												"children": [],
												"opSignDet": "in",
												"rel": " ",
												"isCopy": "",
												"dictionaryCode": [
													"601398.SH",
													"601229.SH",
													"A02105.PF",
													"A02160.PF",
													"A02129.PF",
													"A02132.PF",
													"A02133.PF",
													"A02176.PF"
												]
											}
										]
									}
								],
								"rel": "and",
								"logicalRelation": ""
							}
						],
						"rel": " "
					}
				]
			},
			{
				"tagCodeDet": "NODE_ID",
				"opSignDet": "in",
				"rel": "and",
				"dictionaryCode": [
					"N00000004"
				]
			}
		]}


def update_str(str_need_sub):
    return str_need_sub.replace('\n','').replace('\t','').replace('  ','').replace('   ','')

def expend_shuffle(values_array,repeat_num = 6000000):
    repeat_times = repeat_num/values_array.__len__()
    values_array = np.repeat(values_array,repeat_times)
    np.random.shuffle(values_array)
    return values_array

def generate_test_df():
    detail_asset_code_new_label = [
                            502010201,704010201,701010201,704010101,101010101,
                            104020103,502060101,802010101,806010101,505010101,
                            805030101,805020101,505020101,'C01051001','C01015001',
                            ]

    currency_cd_new_lab = list(range(0,9))+[999]

    is_mezzanine = ['','N','Y']
    is_eliminate =  ['','N','Y']
    special_bussiness_type = [2,3,4,6,16,np.nan]
    currency_cd_new_label = ['1','2','3']
    stanproject_id = [
                    '3001W000000097',
                    '3001W000000041',
                    '3001W000000113',
                    '3001W000000031',
                    '3001W000000108',
                    '3001W000001833',
                    '4001W000000431',
                    '3001PAP0000298',
                    '3001PAP0000315',
                    '3001PAP0000298',
                    '3001PAP0000353',
                    '3001PAP0000407',
                    '3001PAP0000469',
                    '3001PAP0000504',
                    'W000010235',
                    '1001P2017024',
                    '1001P2017026',
                    '1001P2017025',
                    '1001P2017028',
                    '1001P2017027',]

    taa_security_id = ['601398.SH','601229.SH','A02105.PF','A02160.PF','A02129.PF','A02132.PF','A02133.PF','A02176.PF']

    node_id = [
        'N000000002',
        'N000000003',
        'N000000004',
        'N000000005',
        'N000001014',
        'N000002440',
                ]
    code_tier3 = ["20401","50201","50208","50209","50218","50207","50503","50505","50506","50508"]
    sector_level_invest=["D01","K01"]
    arrange_cmb = '3'
    security_id = ['A02856','A03159']
    special_business_type = [str(i) for i in list(range(1,21))]
    world_classing_cd_i9 = [str(i) for i in list(range(1,11))]
    data_matrix = pd.DataFrame()
    data_matrix['node_id'] = expend_shuffle(node_id)
    data_matrix['stanproject_id'] = expend_shuffle(stanproject_id)
    data_matrix['special_bussiness_type'] = expend_shuffle(special_bussiness_type)
    data_matrix['is_mezzanine'] = expend_shuffle(is_mezzanine)
    data_matrix['is_eliminate'] = expend_shuffle(is_mezzanine)
    data_matrix['currency_cd_new_lab'] = expend_shuffle(currency_cd_new_lab)
    data_matrix['detail_asset_code_new_label'] = expend_shuffle(detail_asset_code_new_label)
    data_matrix['currency_cd_new_label'] = expend_shuffle(currency_cd_new_label)
    data_matrix['security_id'] = expend_shuffle(security_id)
    data_matrix['special_business_type'] = expend_shuffle(special_business_type)
    data_matrix['taa_security_id'] = expend_shuffle(taa_security_id)
    data_matrix['world_classing_cd_i9'] = expend_shuffle(world_classing_cd_i9)
    data_matrix['world_classing_cd'] = expend_shuffle(world_classing_cd_i9)
    data_matrix['long_term_invest_type'] = expend_shuffle(world_classing_cd_i9)
    data_matrix['sector_level_invest'] = expend_shuffle(sector_level_invest)
    data_matrix['code_tier3'] = expend_shuffle(code_tier3)
    data_matrix.columns  = [i.upper() for i in data_matrix.columns]

    df = data_matrix
    return df


# ＃.遍历找到字典的所有路径
def dic2sql(node,query_method='sql'):

	if query_method == 'sql':
	# 按照深度优先,先扩到
		if not node.get('children'):

			relation =node.get('rel','')
			node['opSignDet'] = node["opSignDet"].rstrip()

			if node["opSignDet"] == "like" or node["opSignDet"] == 'not like':
				dictionaryCode = f"'{node['dictionaryCode'][0]}'"
			elif node["opSignDet"] == "pre like":
				node["opSignDet"] = "like"
				dictionaryCode = f"'{node['dictionaryCode'][0][2:]}'"
			elif node['opSignDet'] == 'suf like':
				node["opSignDet"] = "like"
				dictionaryCode = f"'{node['dictionaryCode'][0][:-2]}'"
			elif node['opSignDet'] == 'in' or node['opSignDet'] == 'not in':
				dictionaryCode = ','.join(["'%s'" % item for item in node['dictionaryCode']])
				dictionaryCode = f'({dictionaryCode})'

			elif node['opSignDet'] == 'is':
				dictionaryCode = "null"
				return f"{relation} (({node['tagCodeDet']} {node['opSignDet']} {dictionaryCode}) or ({node['tagCodeDet']} = ''))"
			elif node['opSignDet'] == "is not":
				dictionaryCode ="null"
				return f"{relation} (({node['tagCodeDet']} {node['opSignDet']} {dictionaryCode}) or ({node['tagCodeDet']} != ''))"
			# 布尔运算
			elif node['opsignDet'] in ['>','<','>=','<=']:
				dictionaryCode=node['dictionarycode'][0]
				return f"{relation} ({node['tagCodeDet']} {node['opSignDet']} '{dictionaryCode}')"
			else:
				dictionaryCode = node['dictionaryCode']
			# 在这里返回的是正常拼接下的结果
			return f"{relation} ({node['tagCodeDet']} {node['opSignDet']} {dictionaryCode})"

		res= ' '
		tag = node.get('rel','')
		prefix = node.get("prefix")
		suffix = node.get("suffix")
		# 自己调用自己,在这里开分支
		for i in node.get('children'):
			res = f"{res} {dic2sql(i,query_method)}"
		# 如果有的话就对条件取反
		if prefix:
			res = f"{tag} {prefix} ({res}) {suffix}"
		else:
			res = f"{tag} ({res})"
		return res

	if query_method == 'hdfs':
		if not node.get('children'):
			relation = node.get('rel','')
			opSignDet = node['opSignDet']

			if relation.__len__()>0:
				relation = relation.replace('and','&')
				relation = relation.replace('or','|')
			node['opSignDet'] = node["opSignDet"].rstrip()

			# 模糊查询
			if node["opSignDet"] == "like" or node["opSignDet"] == 'not like':
				dictionaryCode = f"'{node['dictionaryCode'][0]}'"

				dictionaryCode = dictionaryCode.replace('%','')
				if opSignDet == 'like':
					return f"{relation} ({node['tagCodeDet']}.str.contains({dictionaryCode}))"
				if opSignDet == 'not like':
					return f"{relation} (~{node['tagCodeDet']}.str.contains({dictionaryCode}))"
			# 开头结尾
			elif node["opSignDet"] == "pre like":
				node["opSignDet"] = "like"
				dictionaryCode = f"'{node['dictionaryCode'][0][2:]}'"
				return f"{relation} ({node['tagCodeDet']}.str.startswith({dictionaryCode})"
			elif node['opSignDet'] == 'suf like':
				node["opSignDet"] = "like"
				dictionaryCode = f"'{node['dictionaryCode'][0][:-2]}'"
				return f"{relation} ({node['tagCodeDet']}.str.endswith({dictionaryCode})"

			# 列表判断
			elif node['opSignDet'] == 'in' or node['opSignDet'] == 'not in':
				dictionaryCode = ','.join(["'%s'" % item for item in node['dictionaryCode']])
				dictionaryCode = f'({dictionaryCode})'
			#
			elif node['opSignDet'] == 'is':
				dictionaryCode = "null"
				return f"{relation} (  ({node['tagCodeDet']}.isnull() )| ({node['tagCodeDet']} == '')  )"
			elif node['opSignDet'] == "is not":
				dictionaryCode ="np.nan"
				return f"{relation} (  ~({node['tagCodeDet']}.isnull() )| ({node['tagCodeDet']} != '')  )"

			# 布尔运算[纯粹的数值运算]
			elif node['opsignDet'] in ['>','<','>=','<=']:
				dictionaryCode=node['dictionarycode'][0]
				return f"{relation} ({node['tagCodeDet']} {opSignDet} '{dictionaryCode}')"
			else:
				dictionaryCode = node['dictionaryCode']
			return f"{relation} ({node['tagCodeDet']} {opSignDet} {dictionaryCode})"

		res= ' '
		tag = node.get('rel','')
		if tag.__len__()>0:
			tag = tag.replace('and','&')
			tag = tag.replace('or','|')
		prefix = node.get("prefix")
		suffix = node.get("suffix")
		# 自己调用自己,在这里开分支
		for i in node.get('children'):
			res = f"{res} {dic2sql(i,query_method)}"
		# 如果有的话就对条件取反
		if prefix:
			res = f"{tag} (~({res}))"
		else:
			res = f"{tag} ({res})"
		return res


from test_json_params import finally_sql_json
node = finally_sql_json




# 把查询捞出来,让后再把把里面的str_修改下
res = dic2sql(node,'hdfs')
res_update = update_str(res)

# df就是取出来的数据
df = generate_test_df()

t0 =time.time()
target_df = df.query(res_update,engine = 'python')
t1 = time.time()
print(t0-1)
print(res_update)
print(target_df)


# start_with query 和 end_with query
# df.query('DETAIL_ASSET_CODE_NEW_LABEL.str.contains('N')',engine = 'python')

# store_client.select('test','''df.NODE_ID == df[df.NODE_ID.str.contains('N')] ''',engine='pandas')
# df.query('''NODE_ID.str.contains('N')''',engine = 'python')
