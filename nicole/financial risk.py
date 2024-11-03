import pandas_datareader
import matplotlib.pyplot as plt
import numpy as np

#sp500 price
sp500=pandas_datareader.data.DataReader(['sp500'],data_source='fred',start='12-30-2012',end='12-30-2022')
#plot sp500 price
plt.plot(sp500["sp500"],color='dodgerblue')


df=pandas_datareader.data.DataReader(['sp500'],data_source='fred',start='12-28-2010',end='12-28-2022')
df.dropna(inplace=True)

#daily log return
df['Daily return squared']=np.log(df['sp500'])/df['sp500'].shift(1)*np.log(df['sp500']/df['sp500'].shift(1))
df.dropna(inplace=True)

#calculate simple moving average
win_list=[5,50,100,250]
for win in win_list:
    ma=df['sp500'].rolling(win).std()
    df[win]=ma
    df.rename(columns={win:'Vol via'+str(win)+"days MA"},inplace=True)

#plot dataframe
flg,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(12,12))
#sp500 price
ax1.plot(df['sp500'])
ax1.set_title('sp500 price')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')
#daily log return squared
ax2.set_title(df['Daily return squared'])
ax2.set_xlabel('Date')
ax2.set_ylabel('Daily return squared')
ax2.spines['right'].set_visible(False)
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')
#ma vol
ax3.plot(df.loc[:,(df.colu,ms!='sp300')&(df.columns!='Daily return squared')])
ax3.legend(df.loc[:,(df.colunns!='sp500')&df.columns!='Daily return squared'].columns)
ax3,set_title('sp500 price volatility via mobing average analysis')
ax3.set_xlabel('Date')
ax3.set_ylabel('Volatility')
ax3.spines['right'].set_visible(False)ssssssssssssssssssssssssssssssss
ax3.yaxis.set_ticks_position('left')
ax3.xaxis.set_ticks_position('bottom')
fig.tight_layout()
