import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#from scipy.stats import zscore

#esempio corretto di read_csv con parse dates
#sales = pd.read_csv('sales.csv', index_col='Date', parse_dates=True)

df_csv=pd.read_csv('Estrazione_CD_fin.csv',sep=';',parse_dates=['TRACK_IN_TIME','TRACK_OUT_TIME','START_TIME','Value_Time'],infer_datetime_format=True,index_col='START_TIME')
df_csv.dropna(inplace=True)
df_csv.Value=df_csv[['Value']].apply(pd.to_numeric, errors='coerce')
df_csv['DES_ID']=df_csv.PRODUCT_ID.str[:4]
print(df_csv.info())
print(df_csv.head(100))
'''
#df_cs_fin è una selezione solo per alcuni des_id
#product_list='C25KA09AAA'
#df_csv_fin=df_csv.loc[df_csv.PRODUCT_ID.str.contains(product_list)]
des_list=['C25K','C24K','C24A']
df_csv_fin=df_csv.loc[df_csv.DES_ID.isin(des_list)]
print('ecco il df per alcune devices')
print(df_csv_fin.head(10))

#boxplot CD by ChamberId
df_csv_fin.boxplot(by='CHAMBER_ID',column ='Value', grid = False,patch_artist=True,
            # Set facecolor to red
            boxprops=dict(facecolor='r'))
plt.title('CD to Chamber_ID')
plt.xlabel('Etch Chamber')
plt.ylabel('um')
plt.show()

#boxplot CD by Device and ChamberId
df_csv_fin.boxplot(by=['DES_ID','CHAMBER_ID'],column ='Value', grid = False)
plt.xlabel('Etch Chamber')
plt.ylabel('um')
plt.show()
'''

#Data Analysis
#proviamo a raggruppare by Des_Id e By Chamber_Id calcolando la media delle CD.
df_csv_des_ch=df_csv.groupby(['DES_ID','CHAMBER_ID'])['Value'].mean()
#df_csv_des_ch=df_csv.groupby('DES_ID')['Value'].mean()
print('Mean by des_id and Chambers')
print(df_csv_des_ch.head())

#invece ora creo semplicemente un dataframe multiindice
df_csv_2ind=df_csv.set_index(['DES_ID','CHAMBER_ID'])
print(df_csv_2ind.head(30))
#Sappiamo però che matplotlib ha problemi a plottare dataframe multi indice
#la soluzione è usare stack()/unstack() ma se ne l dataframe ci sono duplicates entries -> pivot_table
#df_csv=df_csv.unstack(level='CHAMBER_ID')
#print(df_csv.head(30))
df_csv_1=df_csv.pivot_table(index='DES_ID',columns='CHAMBER_ID',values='Value',aggfunc='mean')
print(df_csv_1.head(10))
# plot with matplotlib
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax=df_csv_1.boxplot(patch_artist=True)
plt.show()
# prova di boxplot with data (seaborn)
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
vals, names, xs = [],[],[]
for i, col in enumerate(df_csv_1.columns):
    vals.append(df_csv_1[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, df_csv_1[col].values.shape[0]))
ax = df_csv_1.boxplot(vals, labels=names)
palette = ['r', 'g', 'b', 'y']
for x, val, c in zip(xs, vals, palette):
    ax = df_csv_1.scatter(x, val, alpha=0.4, color=c)
plt.show()


df_csv_2=df_csv.pivot_table(index=['DES_ID','CHAMBER_ID'],values='Value',aggfunc='mean')
print(df_csv_1.head(30))
print('prova')
print(df_csv_2.head(20))

#questo boxplot non sta funzionando come vorrei
df_csv_2.boxplot(patch_artist=True)
plt.show()




#voglio sapere qual'è la devices che risulta con la media sulle CD separate rispetto alle altre (vedi boxplot)
df_csv_des=df_csv.groupby('DES_ID')['Value'].mean()
df_csv_des=df_csv_des.sort_values(ascending=False)
print(df_csv_des.head(15))
#Dovremmo creare delle carte separate per: K22J,K22B,K22F e C24G !!
#Zscore (in realtà è una zscore senza distinguere le camere)
by_month_mean=df_csv.groupby(df_csv.index.strftime('%m'))['Value'].mean()
print(by_month_mean.head(10))
def zscore(series):
    return (series - series.mean())/series.std()
#creo un raggruppamento mensile
by_month=df_csv.groupby([df_csv.index.strftime('%m')])
zscore_by_month=by_month['Value'].transform(zscore)
print(zscore_by_month.head(10))
