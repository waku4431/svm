#ライブラリインポート
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC

#教師データ読み込み
df=pd.read_csv('kyousi1.csv')

#教師データからトレーニングに使いたい列だけを抽出（.valuesはnumpyに変換する奴）
x=df.drop(['半径','x座標','t'],axis=1).values

#目的値をnumpyの型で抽出
t=np.array(df['t'])

# 機械学習させる
clf = SVC(kernel='linear',C=1.0)
clf.fit(x, t)

for i in range(10):
  plt.scatter(df[df.t==i]['No'],df[df.t==i]['y座標'])
plt.autoscale()
plt.grid()
plt.show()

#未知データの読み込み
cl=pd.read_csv('miti1.csv')
cl.head()

#未知データの中から予測に用いる値だけを抽出
x2=cl.drop(['半径','x座標'],axis=1).values

#未知データを学習モデルで分類する
x2_run=clf.predict(x2)#学習モデルで予測
x3=pd.DataFrame(x2_run)#pandasに変換
x3.columns=['t']#列名取得

#元データと予測したデータを見やすいように結合する
fi=cl.join(x3)#cl+x3を横に結合

for i in range(10):
  plt.scatter(fi[fi.t==i]['No'],fi[fi.t==i]['y座標'])
plt.autoscale()
plt.grid()
plt.show()

#データをcsvファイルに保存
fi.to_csv('output.csv')