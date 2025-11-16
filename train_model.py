import pandas as pd, joblib, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE=os.path.dirname(__file__)
CSV=os.path.join(BASE,'sample_transactions.csv')
MODEL=os.path.join(BASE,'saved_models','cat_model.pkl')
TFIDF=os.path.join(BASE,'saved_models','tfidf.pkl')
df=pd.read_csv(CSV).dropna()
tf=TfidfVectorizer(max_features=1500)
X=tf.fit_transform(df['description']); y=df['category']
model=LogisticRegression(max_iter=200).fit(X,y)
joblib.dump(tf,TFIDF); joblib.dump(model,MODEL)
print("trained")