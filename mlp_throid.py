import pandas as pd
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')






df = pd.read_csv("allrep.csv")
print(df.info())



df.head()

# noktalamalar için
df['Class'] = df['Class'].str.replace('[^\w\s]','')

#sayılar
df['Class'] = df['Class'].str.replace('\d','')

df["Class"].value_counts() 

df["TBG"].value_counts() # hepsi nan value drop edilmeli

df.columns # columns names

df.shape # shape of dataframe

df.isnull().sum() # missing datas

# some data normally can be numerical type for ex:age,TSH,T3,TT4,T4U,FTI,TBG 
#but all data seems to object type
#This problem due to '?' characters
#Convert to this character np.nan

df.dtypes 

# replace to '?' to np.nan
# so real missing values can be count
mymap = {"?":np.NaN}

df=df.applymap(lambda s: mymap.get(s) if s in mymap else s)

df.isnull().sum() # missing datas can be observed

df[['sex']] = df[['sex']].replace(to_replace={'F':1,'M':0})

df[['on_thyroxine','query_on_thyroxine','on_antithyroid_medication','lithium','goitre','tumor','hypopituitary','psych']] = df[['on_thyroxine','query_on_thyroxine','on_antithyroid_medication','lithium','goitre','tumor','hypopituitary','psych']].replace(to_replace={'t':1,'f':0})

df[['TSH_measured','T3_measured','TT4_measured','T4U_measured','FTI_measured','TBG_measured','referral_source']] = df[['TSH_measured','T3_measured','TT4_measured','T4U_measured','FTI_measured','TBG_measured','referral_source']].replace(to_replace={'t':1,'f':0})

df[['sick','pregnant','thyroid_surgery','I131_treatment','query_hypothyroid','query_hyperthyroid']] = df[['sick','pregnant','thyroid_surgery','I131_treatment','query_hypothyroid','query_hyperthyroid']].replace(to_replace={'t':1,'f':0})

df.head()


df.dtypes


df.isnull().sum()

df["age"] = df["age"].astype(float)
df["TSH"] = df["TSH"].astype(float)
df["T3"] = df["T3"].astype(float)
df["TT4"] = df["TT4"].astype(float)
df["T4U"] = df["T4U"].astype(float)
df["FTI"] = df["FTI"].astype(float)


df.dtypes

for i in ['age','sex','TSH','T3','TT4','T4U','FTI']:
    df[i].fillna(df[i].mean(),inplace=True)




df.isnull().sum()

df['Class'].value_counts() # 4 output



y= df['Class']

x_data = df.drop(['referral_source','Class','TBG'],axis=1)


# %% normalization
from sklearn.preprocessing import StandardScaler
Scaler=StandardScaler()
x=Scaler.fit_transform(x_data)

#%%  keras
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y = onehot_encoder.fit_transform(integer_encoded)

y[0:3]

#%%

from sklearn.model_selection import train_test_split
# shuffle and split training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0) #test_size=0.4,0.33,random_state=1,10,20 

from sklearn.neural_network import MLPClassifier 

mlpc = MLPClassifier() 

mlpc.fit(x_train, y_train) 



y_pred = mlpc.predict(x_test) 
#%% accuracy
import sklearn.metrics as metrics
print(metrics.accuracy_score(y_test, y_pred))

#%% grid search 
    
mlpc_params = {"alpha": [0.1, 0.01, 0.0001],
              "hidden_layer_sizes": [(10,10,10),
                                     (100,100,100), #fazladan parametreler kaldırıldı ram kaynaklı çalışma sorunu yaşadım
                                     (100,100),],
              "solver" : ["lbfgs","adam","sgd"],
              "activation": ["relu","logistic"]}

from sklearn.model_selection import GridSearchCV




mlpc = MLPClassifier() 

mlpc_cv_model = GridSearchCV(mlpc, mlpc_params, 
                         cv = 5, 
                         n_jobs = -1,
                         verbose = 2) 

mlpc_cv_model.fit(x_train, y_train) 


print(mlpc_cv_model.best_params_)
#%%
mlpc_tuned = MLPClassifier(activation = "logistic", 
                           alpha = 0.01, 
                           hidden_layer_sizes = (100, 100),
                          solver = "lbfgs")




# %%
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
clf = MLPClassifier(solver='lbfgs',activation = "logistic", alpha=0.01, hidden_layer_sizes=(100,100), random_state=0)

for train_indices, test_indices in kf.split(x):
    clf.fit(x[train_indices], y[train_indices])
    print(clf.score(x[test_indices], y[test_indices]))
#%%
mlpc_tuned.fit(x_train, y_train)



# Final Modelinin test seti üzerinden tahmin işlemi
y_pred = mlpc_tuned.predict(x_test)


# %%
print("acc:",metrics.accuracy_score(y_test,y_pred))
# %%
# %%
#confusion matrix
print(metrics.confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))

# %%
print("f1:",metrics.f1_score(y_test,y_pred,average="micro"))

# %%
print(metrics.classification_report(y_test,y_pred))


#%%
import matplotlib.pyplot as plt
from scipy import interp

from sklearn.metrics import roc_curve, auc
from itertools import cycle




y_score = mlpc_tuned.predict_proba(x_test)


n_classes = 4 # output sayısı

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0]) # 0. sınıf için
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


#%%
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()