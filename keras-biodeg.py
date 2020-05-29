import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


from keras.models import Sequential
from keras.layers import Dense,Activation
df = pd.read_csv("qsar-biodeg.csv")


df["Class"].value_counts() 
# eady biodegradable (RB) and not ready biodegradable (NRB)
# RB:2 NRB:1

df.Class=[1 if each ==2 else 0 for each in df.Class]

df["Class"].value_counts() 
# eady biodegradable (RB) and not ready biodegradable (NRB)
# RB:1 NRB:0

y = df["Class"].values
x_data = df.drop(['Class'], axis=1)

# %% normalization
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.4,random_state=42) #test_size=0.3,0.33,random_state=1,10,20 

#%% keras


def keras_model(optimizer="adam"):
    model=Sequential()
    model.add(Dense(16,input_dim=41))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(optimizer=optimizer,loss="binary_crossentropy",metrics=["accuracy"])
    return model


model = keras_model() 

egitim=model.fit(x_train, y_train, epochs=100,validation_data=(x_test,y_test))
# %% plot loss during training

plt.plot(egitim.history['loss'], label='train')
plt.plot(egitim.history['val_loss'], label='test')
plt.title('model loss')
plt.xlabel('epochs')
plt.ylabel('loss values')
plt.legend(loc='upper right')
plt.show()



# %%
import sklearn.metrics as metrics
y_pred=model.predict_classes(x_test)
# %%Accuracy

print("Accuracy:",metrics.accuracy_score(y_test,y_pred))

# %%f1 score

print("f1:",metrics.f1_score(y_test, y_pred))


#%% Grid Search 

keras_param = {
   
    'epochs': [100,150,200],

    'optimizer':['RMSprop', 'Adam','SGD'],
    
}

keras_cl = KerasClassifier(build_fn=keras_model, verbose=1)


keras_cv = GridSearchCV(estimator=keras_cl,  
                    n_jobs=-1, 
                    verbose=1,
                    cv=5,
                    param_grid=keras_param)

keras_cv_model = keras_cv.fit(x_train, y_train,) 

#%%

print(keras_cv_model.best_params_)

#%%

def keras_cv_model():
    model=Sequential()
    model.add(Dense(16,input_dim=41))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(optimizer='RMSprop',loss="binary_crossentropy",metrics=["accuracy"])
    return model


keras_model_kf=KerasClassifier(build_fn=keras_cv_model,epochs=200,verbose=1)

#%% k-fold

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


kf = KFold(n_splits=5, shuffle=True,random_state=42)
accuracies = cross_val_score(keras_model_kf, x_test, y_test, cv=kf,scoring= 'accuracy')


print(' Accuracies ', accuracies)
#%%
keras_model_tuned=keras_cv_model()
keras_model_tuned.fit(x_train,y_train,epochs=200)

# %% cm
import sklearn.metrics as metrics
y_pred=keras_model_tuned.predict(x_test)
# %%
print("acc:",metrics.accuracy_score(y_test,y_pred.round()))
# %%
print("cm:",metrics.confusion_matrix(y_test,y_pred.round()))

# %%
print("f1:",metrics.f1_score(y_test,y_pred.round()))

# %%
print(metrics.classification_report(y_test,y_pred.round()))
# %% roc ve auc
import matplotlib.pyplot as plt

probs=keras_model_tuned.predict(x_test)
fpr,tpr,threshold=metrics.roc_curve(y_test,probs)
            

roc_auc=metrics.auc(fpr,tpr)

plt.title("ROC")
plt.plot(fpr,tpr,label="AUC=%0.2f" %roc_auc)
plt.legend(loc="lower right")
plt.ylabel("TPR")
plt.xlabel("FPR")
plt.show()