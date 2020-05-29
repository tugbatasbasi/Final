from warnings import filterwarnings
filterwarnings('ignore') #asla eşleşen uyarıları yazdırmayın
import pandas as pd

df = pd.read_csv("qsar-biodeg.csv") #dataseti oku
print(df.info())


df["Class"].value_counts() #1 ve 2 değerlerini say
# eady biodegradable (RB) and not ready biodegradable (NRB)
# RB:2 NRB:1

df.Class=[1 if each ==2 else 0 for each in df.Class] #2 değerlerini 1 yap diğerlerini 0

df["Class"].value_counts()  #1 ve 0 değerlerini say
# eady biodegradable (RB) and not ready biodegradable (NRB)
# RB:1 NRB:0

y = df["Class"].values #Class sütununu y'ye gönder
x_data = df.drop(['Class'], axis=1) #Class'ı drop et df yi x_data'ya gönder
#%% Veri ölçekleme
from sklearn.preprocessing import StandardScaler #diğerlerinden daha iyi sonuç verdi
Scaler=StandardScaler()
x=Scaler.fit_transform(x_data)
# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.4,random_state=42) #test_size=0.3,0.33,random_state=1,10,20 

#%%
from sklearn.neural_network import MLPClassifier

mlpc_model=MLPClassifier(random_state=0)
mlpc_model.fit(x_train,y_train)

# %%
print("score:",mlpc_model.score(x_test,y_test))

# %% model tuning- GridSearchCV


from sklearn.model_selection import GridSearchCV

mlpc_params = {"alpha": [ 0.001,0.0001],
              "hidden_layer_sizes": [(10,10),
                                     (100,100,100)],   #fazladan parametreler kaldırıldı ram kaynaklı çalışma sorunu yaşadım
              "solver" : ["adam","sgd"],
              "activation": ["relu","logistic"]
              }
            
# mlpc=MLPClassifier() #ilk çalışan
mlpc=MLPClassifier(random_state=0) #2. çalışan
mlpc_cv_model=GridSearchCV(mlpc,mlpc_params,cv=5,n_jobs=-1,verbose=2).fit(x_train,y_train)
# %% Validasyon Yöntemi- KFold
from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
clf = MLPClassifier(solver='lbfgs', alpha=0.0001, hidden_layer_sizes=(100,100,100), random_state=0)

for train_indices, test_indices in kf.split(x):
    clf.fit(x[train_indices], y[train_indices])
    print(clf.score(x[test_indices], y[test_indices]))


# %%
print(mlpc_cv_model.best_params_)
# %%

mlpc_tuned=mlpc_cv_model.best_estimator_.fit(x_train,y_train)
# %%
print("score:",mlpc_tuned.score(x_test,y_test))
# %% cm
import sklearn.metrics as metrics
y_pred=mlpc_cv_model.predict(x_test)
# %%
print("acc:",metrics.accuracy_score(y_test,y_pred))
# %%
print("cm:",metrics.confusion_matrix(y_test,y_pred))

# %%
print("f1:",metrics.f1_score(y_test,y_pred))

# %%
print(metrics.classification_report(y_test,y_pred))
# %% roc ve auc
import matplotlib.pyplot as plt
probs=mlpc_cv_model.predict(x_test)
fpr,tpr,threshold=metrics.roc_curve(y_test,probs)
            
roc_auc=metrics.auc(fpr,tpr)

plt.title("ROC")
plt.plot(fpr,tpr,label="AUC=%0.2f" %roc_auc)
plt.legend(loc="lower right")
plt.ylabel("TPR")
plt.xlabel("FPR")
plt.show()