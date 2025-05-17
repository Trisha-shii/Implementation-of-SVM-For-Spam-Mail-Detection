# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages.
2.Analyse the data.
3.Use modelselection and Countvectorizer to preditct the values.
4.Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: TRISHA PRIYADARSHNI PARIDA
RegisterNumber:  212224230293
*/


import pandas as pd
from google.colab import files
uploaded = files.upload()

data = pd.read_csv('spam.csv',encoding='Windows-1252')

data

----

data.shape

------



x=data['v2'].values
y=data['v1'].values
x.shape

------

y.shape
-----


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

-----
x_train.shape
------
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

------

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
------

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc
------
con=confusion_matrix(y_test,y_pred)
print(con)
-------


cl=classification_report(y_test,y_pred)
print(cl)
----------


```

## Output:
![Screenshot 2025-05-17 204227](https://github.com/user-attachments/assets/ed44a6a5-423c-4663-bfba-e34414af5be0)

![Screenshot 2025-05-17 204318](https://github.com/user-attachments/assets/1d24253c-0a76-46b7-a518-91659335a17d)

![Screenshot 2025-05-17 204424](https://github.com/user-attachments/assets/08c314b7-f2d3-43bf-b980-a7879c7ebe61)

![Screenshot 2025-05-17 204511](https://github.com/user-attachments/assets/3d5f24a1-2978-4f89-9440-35a46c986e0c)

![Screenshot 2025-05-17 204550](https://github.com/user-attachments/assets/414b3dc3-a513-47f5-9dfe-88577f9c8069)

![Screenshot 2025-05-17 204624](https://github.com/user-attachments/assets/d6a15ac4-dc49-478c-9f44-c0eb17a40ce4)

![Screenshot 2025-05-17 204728](https://github.com/user-attachments/assets/782fae15-8f19-4d7d-bf3f-0ac1635367ff)

![Screenshot 2025-05-17 204754](https://github.com/user-attachments/assets/9e6c0b78-e5a4-4230-b417-35b70cc1b602)

![Screenshot 2025-05-17 204826](https://github.com/user-attachments/assets/d04c550d-252f-41e4-acbf-045bf35380c8)

![Screenshot 2025-05-17 204902](https://github.com/user-attachments/assets/98e1c7d1-8d82-4466-8038-2c98b00aa4e1)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
