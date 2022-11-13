#import libraries
import streamlit as st
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

#loading dataset
df=pd.read_csv('diabetes.csv')

st.title("Diabetes prediction app")
st.sidebar.header("patient data")
st.subheader("description stats of data")

st.write(df.describe())

#split into X, y and train test split
X=df.drop(["Outcome"], axis=1)
y=df.iloc[:,-1]

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=0)

#Function
def user_report():
    pregnancies=st.sidebar.slider("Pregnancies", 0,17,2)
    glucose=st.sidebar.slider("Glucose", 0,199,110)
    bp=st.sidebar.slider("BloodPressure",0,122,80)
    sk=st.sidebar.slider("SkinThickness", 0,99,12)
    insulin=st.sidebar.slider("Insulin", 0,846,80)
    BMI=st.sidebar.slider("BMI", 0,67,5)
    dpf=st.sidebar.slider("DiabetesPedigreeFunction", 0.07,2.42,0.37)
    age=st.sidebar.slider("Age", 21,81,33)
    user_report_data={
    "pregnancies":pregnancies,
    "glucose":glucose,
    "bp":bp,
    "sk":sk,
    "insulin":insulin,
    "BMI":BMI,
    "dpf":dpf,
    "age":age}
    report_data=pd.DataFrame(user_report_data, index=[0])
    return report_data

#patient data
user_data=user_report()
st.subheader("patient data")
st.write("user_data")

#model
rc=RandomForestClassifier()
rc.fit(X_train, y_train)
user_result=rc.predict(user_data)

#visualisation
st.title("visualised patient data")

#color funstion
if user_result[0]==0:
    color="blue"
else:
    color="red"

#Age vs Pregnancies
st.header("AGE VS PREGNANCY")
fig_preg=plt.figure()
ax1=sns.scatterplot(x="Age", y="Pregnancies", data=df, hue="Outcome")
ax2=sns.scatterplot(x=user_data["age"],y=user_data["pregnancies"], s=150, color=color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title("0-Healthy & 1-Diabetic")
st.pyplot(fig_preg)

#Age vs Glucose
st.header("AGE VS GLUCOSE")
fig_preg=plt.figure()
ax1=sns.scatterplot(x="Age", y="Glucose", data=df, hue="Outcome")
ax2=sns.scatterplot(x=user_data["age"],y=user_data["glucose"], s=150, color=color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,198,20))
plt.title("0-Healthy & 1-Diabetic")
st.pyplot(fig_preg)

#Age vs Bloodpressure
st.header("AGE VS BLOODPRESSURE")
fig_preg=plt.figure()
ax1=sns.scatterplot(x="Age", y="BloodPressure", data=df, hue="Outcome")
ax2=sns.scatterplot(x=user_data["age"],y=user_data["bp"], s=150, color=color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,122,10))
plt.title("0-Healthy & 1-Diabetic")
st.pyplot(fig_preg)

#Age vs skinthickness
st.header("AGE VS SKINTHICKNESS")
fig_preg=plt.figure()
ax1=sns.scatterplot(x="Age", y="SkinThickness", data=df, hue="Outcome")
ax2=sns.scatterplot(x=user_data["age"],y=user_data["sk"], s=150, color=color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,100,5))
plt.title("0-Healthy & 1-Diabetic")
st.pyplot(fig_preg)

st.bar_chart(df['BMI'])
st.line_chart(df['Insulin'])

#output
st.header("Your Report: ")
Output=" "
if user_result[0]==0:
    Output="YOU ARE HEALTHY😍"
    st.balloons()
else:
    Output="YOU ARE DIABETIC😢"
    st.warning("sugar, sugar, sugar")
st.title(Output)


#accuracy, recall, precision and confusion matrix
rc.fit(X_train,y_train)
accuracy=rc.score(X_test, y_test)
y_pred= rc.predict(X_test)
st.write("Accuracy:", accuracy.round(2))
st.write("Precision:", precision_score(y_test, y_pred))
st.write("Recall:", recall_score(y_test, y_pred))
cm=st.write("confusion matrix:", metrics.confusion_matrix(y_test, y_pred))





    