#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pylab as pl
import pandas as pd 
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[4]:


import os
os.chdir("C:\\Users\\HP\\Desktop")

cell_df = pd.read_csv("cell_samples.csv")
cell_df


# In[6]:


cell_df["Class"] = cell_df["Class"].replace([2],0) # Tumor benigno.
cell_df["Class"] = cell_df["Class"].replace([4],1)# Tumor Maligno
cell_df


# In[12]:


ax = cell_df[cell_df["Class"] ==1 ][0:50].plot(kind = "scatter", x ="Clump", y="UnifSize", color = "DarkRed", label="Maligno");
ax = cell_df[cell_df["Class"] ==0 ][0:50].plot(kind = "scatter", x ="Clump", y="UnifSize", color = "Yellow", label="Maligno", ax= ax);
plt.show()


# In[13]:


# pre_procesamiento de datos y seleción
cell_df.dtypes


# In[14]:


# Eliminación de renglones no numéricos en BareNuc
cell_df = cell_df[pd.to_numeric(cell_df["BareNuc"], errors = "coerce").notnull()]
cell_df["BareNuc"] = cell_df["BareNuc"].astype("int")
cell_df.dtypes


# In[17]:


feature_df = cell_df[["Clump","UnifSize","UnifShape","MargAdh","SingEpiSize","BareNuc","BlandChrom","NormNucl","Mit"]]
x = np.asarray(feature_df)
x[0:5]


# In[19]:


cell_df["Class"] = cell_df["Class"].astype("int")
y = np.asarray(cell_df["Class"])
y[0:5]


# In[20]:


# Creación de grupos de entrenamiento y prueba.
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=4)
print("Grupo de entrenamiento", x_train.shape, y_train.shape)
print("grupo de prueba", x_test.shape, y_test.shape)


# Modelacion con svm

# Opciones de kernel(Transformaciónes)
# 1.Linear
# 2.Polynomial
# 3.Radial Basis Function (RBF)
# 4.Sigmoid

# In[22]:


from sklearn import svm
clf = svm.SVC(kernel = "rbf")
clf.fit(x_train,y_train)


# In[23]:


#Predicción para la base de prueba
yhat = clf.predict(x_test)
yhat[0:5]


# In[24]:


# Creasión de matriz de confución
from sklearn.metrics import confusion_matrix
y_pred = clf.predict(x)
cm = confusion_matrix(y_test,yhat)
cm


# In[25]:


y_test


# In[26]:


yhat


# In[28]:


z = y_test - yhat
z


# In[33]:


# Vizualisación de matriz de confusión
import seaborn as sns
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax= ax)
plt.xlabel("y pronosticada")
plt.ylabel("y real")
plt.show()


# Estadisticas de desempeño

# In[37]:


from sklearn.metrics import classification_report
cnf_metrix = confusion_matrix(y_test, yhat, labels=[0,1])
print(classification_report(y_test,yhat))


# # Determinasión de niveles de presición
# la precision = Porcentaje de predicciones correctas realtivas al total de predicciones
# * El total se calcular por colmna

# * Se contesta a la pregunta: ¿Qué porcentaje de veces que hacemos un pronóstico de cierto tipo éste es correcto?

# In[38]:


precisionbenigno = cm[0,0] / (cm[0,0] + cm[1,0])
precisionbenigno


# In[39]:


precisionmaligno = cm[1,1] / (cm[1,1] + cm[0,1])
precisionmaligno 


# # Determinación de niveles de Recuperación(Recall)
# La recuperacion = Porcentaje de predicciones correctas relativas al total de valores reales
# * El total se calcular por renglon
# * Se contesta a la pregunta: ¿Qué porcentaje de las veces que se tiene un valor es de identificarlo?

# In[40]:


recallbenigno = cm[0,0] / (cm[0,0] + cm[0,1])
recallbenigno


# In[42]:


recallmaligno = cm[1,1] / (cm[1,0] + cm[1,1])
recallmaligno 


# # Determinación del Score F-1
# F1 score = es la media armonica ponderada de la precisión y la recuperación. Cuanto más se acerque a 1 mejor será el modelo.
# F1 score = 2*(Precisión*Recall)/(precisión + Recall)

# In[44]:


f1Benigno = 2* (precisionbenigno * recallbenigno) / (precisionbenigno + recallbenigno)
f1Benigno


# In[45]:


f1Maligno = 2* (precisionmaligno * recallmaligno) / (precisionmaligno + recallmaligno)
f1Maligno


# # Determinación de Soportes
# Soporte = Número de observaciones que pertenecen en forma real a cada clase posible(Total de renglon)

# In[46]:


soportebenigno = cm[0,0] + cm[0,1]
soportebenigno


# In[47]:


soportemaligno = cm[1,0] + cm[1,1]
soportemaligno


# # Determinación de Precisión Global(Accuracy)
# Accuracy = Porcentaje de predicciones correctas.

# In[48]:


correctos = cm[0,0] + cm[1,1]
incorrectos = cm[0,1] + cm[1,0]
PrecisiónGlobal = correctos / (correctos + incorrectos)
PrecisiónGlobal


# # Promedios simples por indicador ( Macro Average)
# Promedio simples por Precisión, Recuperación y F1 score

# In[49]:


MacroAvPre = (precisionbenigno + precisionmaligno) / 2
MacroAvPre


# In[50]:


MacroAvReC = (recallbenigno + recallmaligno) / 2
MacroAvReC


# In[52]:


MacroF1S = (f1Benigno + f1Maligno) / 2
MacroF1S


# # Promedios ponderados por indicador(Weight Average)
# Promedios ponderados de acuerdo al soporte por indicador

# In[53]:


pesobenigno = soportebenigno / (soportebenigno + soportemaligno)
pesomaligno = soportemaligno / (soportebenigno + soportemaligno)


# In[54]:


WAvgPrecision = precisionbenigno * pesobenigno + precisionmaligno * pesomaligno
WAvgPrecision


# In[55]:


WAvgRecall = recallbenigno * recallbenigno + recallmaligno * recallmaligno
WAvgRecall


# In[57]:


WAvgf1 = f1Benigno * f1Benigno + f1Maligno * f1Maligno
WAvgf1


# # Prueba de svm con Kernel Lineal

# In[58]:


clf = svm.SVC(kernel = "linear")
clf.fit(x_train, y_train)
yhat = clf.predict(x_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
y_pred = clf.predict(x)

cm= confusion_matrix(y_test, yhat)
cm


# In[59]:


from sklearn.metrics import classification_report
cnf_metrix = confusion_matrix(y_test, yhat, labels=[0,1])

print(classification_report(y_test,yhat))


# # Prueba de SVM con kernel polinomial
# 

# In[60]:


clf = svm.SVC(kernel = "poly")
clf.fit(x_train, y_train)
yhat = clf.predict(x_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
y_pred = clf.predict(x)

cm= confusion_matrix(y_test, yhat)
cm


# In[61]:


from sklearn.metrics import classification_report
cnf_metrix = confusion_matrix(y_test, yhat, labels=[0,1])

print(classification_report(y_test,yhat))


# # Prueba de SVM con Kernel Sigmoide

# In[62]:


clf = svm.SVC(kernel = "sigmoid")
clf.fit(x_train, y_train)
yhat = clf.predict(x_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
y_pred = clf.predict(x)

cm= confusion_matrix(y_test, yhat)
cm


# In[63]:


from sklearn.metrics import classification_report
cnf_metrix = confusion_matrix(y_test, yhat, labels=[0,1])

print(classification_report(y_test,yhat))


# In[ ]:




