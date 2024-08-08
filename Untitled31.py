#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir("C:\\Users\\HP\\Desktop")


# In[4]:


data = pd.read_csv("Breast-cancer.csv")
data


# In[ ]:


data.drop(["id", "Unnamed: 32"], axis = 1, inplace=True)
data.head()


# In[8]:


M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]


# In[9]:


#Diagrama de disperción
plt.scatter(M.radius_mean, M.texture_mean, color = "red", label="Maligno", alpha = 0.3)
plt.scatter(B.radius_mean, B.texture_mean, color = "blue", label="Benigno", alpha = 0.3)
plt.xlabel("Radius_mean")
plt.ylabel("Texture_mean")
plt.legend()
plt.show()


# In[10]:


# 0 = Benigno, 1 = Maligno
data.diagnosis = [1 if each =="M" else 0 for each in data.diagnosis]
data


# In[12]:


y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis = 1)


# In[13]:


y


# In[14]:


x_data


# In[17]:


# Normalización

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
x


# In[18]:


# Bases de entrenamiento y prueba.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30 , random_state= 1)


# In[25]:


#Modelo de KNN (K vecinos mas cercanos)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("Score: ",knn.score(x_test, y_test))


# In[26]:


y_pred = prediction
y_true = y_test


# In[27]:


#Creasión de matirz de confución
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
cm


# In[28]:


# Vizializasión de la matriz de confución
import seaborn as sns
f, ax =plt.subplots(figsize =(5,5))
sns.heatmap(cm, annot =True, linewidths = 0.5, linecolor = "red", fmt = ".0f", ax = ax)
plt.xlabel("y pronosticada")
plt.ylabel("y verdadera")
plt.show()


# In[30]:


# Cálculo de la precisión global
Correctos = cm[0,0] + cm[1,1]
Incorrectos = cm[0,1] + cm[1,0]
PredicciónGlobal = Correctos / (Correctos + Incorrectos)
PredicciónGlobal


# In[31]:


PresiciónBenignos = cm[0,0] / (cm[0,0] + cm[1,0])
PresiciónBenignos


# In[32]:


PresiciónMalignos = cm[1,1] / (cm[1,1] + cm[0,1])
PresiciónMalignos 


# In[33]:


from sklearn.metrics import roc_curve
from sklearn.metrics import auc

y_scores = knn.predict_proba(x_test)
fpr, tpr, threshold = roc_curve(y_test, y_scores[:,1])
#fpr = False positive rate , tpr = True positive rate
roc_auc = auc(fpr,tpr)

plt.title("Reciver Operating Characteristic")
plt.plot(fpr,tpr, "b", label = " AUC = %0.2F" % roc_auc)
plt.legend(loc = "lower right")
plt.plot([0,1],[0,1], "r--")
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel("Tasa de positivos Verdaderos")
plt.xlabel("Tasa de Falsos positivos")
plt.title("Curva de ROC de KNN")
plt.show


# In[ ]:




