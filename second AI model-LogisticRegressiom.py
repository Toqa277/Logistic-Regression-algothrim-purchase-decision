# importing libiraries
# model social network ads classificaion
import pandas
import pandas as pd   # to read data
dataset = pandas.read_csv("Social_Network_Ads.csv")
x= dataset.iloc[: , 2:4].values
y= dataset.iloc[: , 4].values
# scaling (normalization)
from sklearn.preprocessing import StandardScaler # to normalize the dataset
sc= StandardScaler()
x= sc.fit_transform(x) #normalizin x beofre dividing it into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size= 0.30)
from sklearn.linear_model import LogisticRegression # importing the algorithim
classifier = LogisticRegression()
# trainging phase
classifier.fit(x_train, y_train) #model is created
y_predict = classifier.predict(x_test) #testing phase
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict) # to comparse between the reel and predicted values
print(cm )
acuracy= ((cm[0][0])+(cm[1][1]))/(len(y_test)) ; print("accuarcy of Logistc regression model is:", acuracy*100,"%")
