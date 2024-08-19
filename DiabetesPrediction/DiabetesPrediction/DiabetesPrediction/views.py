from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score






def home(request):
    return render(request, 'home.html')
def predict(request):
    return render(request, 'predict.html')
def result(request):
    data = pd.read_csv(r"C:\Users\home\Downloads\archive (1).zip")
    x = data.drop("Outcome", axis=1)
    y = data['Outcome']
    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.30)

    from sklearn.linear_model import LogisticRegression

    # Initialize the Logistic Regression model
    model = LogisticRegression()
    model.fit(x_train,y_train)


    # from sklearn.preprocessing import StandardScaler
    # from imblearn.over_sampling import SMOTE
    #
    # scaler = StandardScaler()
    # x_standardized = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
    #
    # # Data Augmentation: Synthetic Data Generation
    # smote = SMOTE(random_state=42)
    # x_resampled, y_resampled = smote.fit_resample(x_standardized, y)
    #
    # # Split the resampled data into training and testing sets
    # x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)
    #
    # from sklearn.neighbors import KNeighborsClassifier
    #
    # knn_model = KNeighborsClassifier()
    # knn_model.fit(x_train, y_train)
    #


    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

    result1 = ""
    if pred == [1]:
        result1 = "Positive"
    elif pred == [0]:
        result1 = "Negative"

    return render(request, "predict.html", {"result2": result1})