# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 19:08:46 2024

@author: HEMANT
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 15:27:10 2022

@author: HEMANT KUMAR
"""

#import altair as alt
import sklearn
import streamlit as st
import pandas as pd
import warnings
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sn
import matplotlib.pyplot as plt
import plotly.express as px

# Dictionary to store the maximum accuracy for all algorithms
D = {}

# Ignore warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Streamlit app title and sidebar
st.title("Parkinson's Classification using Multiple ML Models")
st.sidebar.title("Model Selection")

# Uploading the CSV file
uploaded_file = st.sidebar.file_uploader("Parkinsons.csv", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Display dataframe
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # Extracting features and labels from the dataset
    features = df.loc[:, df.columns != 'status'].values[:, 1:]
    labels = df.loc[:, 'status'].values

    # Min-Max Normalization
    scaler = MinMaxScaler((-1, 1))
    x = scaler.fit_transform(features)
    y = labels

    # Splitting the dataset into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Sidebar options for model selection
    model_selection = st.sidebar.selectbox("Choose Classifier", ("SVM", "KNN", "Modified KNN", "Random Forest", "GMM1", "GMM2"))

    # Function for SVM
    if model_selection == "SVM":
        st.write("### Classification using SVM")
        from sklearn.svm import SVC
        classifi2 = SVC(kernel='linear')
        classifi2.fit(x_train, y_train)
        y2_pred = classifi2.predict(x_test)

        # Accuracy and confusion matrix
        Acc_Svm = accuracy_score(y_test, y2_pred) * 100
        D["SVM"] = round(Acc_Svm, 3)
        st.write(f"Accuracy: {round(Acc_Svm, 3)}%")

        cm = confusion_matrix(y_test, y2_pred)
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots()
        sn.heatmap(cm, annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)

    # KNN algorithm
    if model_selection == "KNN":
        st.write("### Classification using KNN")
        k = st.sidebar.selectbox("Select K value", [1, 3, 5, 7])

        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        Acc_knn = accuracy_score(y_test, y_pred) * 100
        D["KNN"] = round(Acc_knn, 3)
        st.write(f"Accuracy: {round(Acc_knn, 3)}%")

        cm = confusion_matrix(y_test, y_pred)
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots()
        sn.heatmap(cm, annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)

    # Random Forest algorithm
    if model_selection == "Random Forest":
        st.write("### Classification using Random Forest")
        classifier = RandomForestClassifier(n_estimators=40, max_depth=15, random_state=50)
        classifier.fit(x_train, y_train)
        Y_predict = classifier.predict(x_test)

        accuracy_random = accuracy_score(y_test, Y_predict) * 100
        D["Random Forest"] = round(accuracy_random, 3)
        st.write(f"Accuracy: {round(accuracy_random, 3)}%")

        cm = confusion_matrix(y_test, Y_predict)
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots()
        sn.heatmap(cm, annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)

    # Gaussian Mixture Model (Unimodal)
    if model_selection == "GMM1":
        st.write("### Classification using Unimodal Gaussian Mixture Model")
        df = df.drop('name', axis=1)
        X = df.copy()
        X_label = df['status']
        [X_train, X_test, X_label_train, X_label_test] = train_test_split(X, X_label, test_size=0.3, random_state=42, shuffle=True)
        
        df3 = X_train
        df4 = X_test
        df_0 = df3.loc[df3["status"] == 0, :]
        df_1 = df3.loc[df3["status"] == 1, :]
        df4 = df4.drop('status', axis=1)
        df_0 = df_0.drop('status', axis=1)
        df_1 = df_1.drop('status', axis=1)

        mean_0 = df_0.mean()
        cov_0 = df_0.cov()
        mean_1 = df_1.mean()
        cov_1 = df_1.cov()

        k = []
        for i in range(df4.shape[0]):
            p = np.matmul(np.transpose(df4.iloc[i] - mean_0), np.linalg.inv(cov_0))
            CCD_0 = (np.exp(-(np.matmul(p, df4.iloc[i] - mean_0)) / 2)) / ((2 * np.pi) * (np.linalg.det(cov_0) * 0.5))
            q = np.matmul(np.transpose(df4.iloc[i] - mean_1), np.linalg.inv(cov_1))
            CCD_1 = (np.exp(-(np.matmul(q, df4.iloc[i] - mean_1)) / 2)) / ((2 * np.pi) * (np.linalg.det(cov_1) * 0.5))

            PC_0 = len(df_0) / len(df3)
            PC_1 = len(df_1) / len(df3)
            P_0 = (CCD_0 * PC_0) / (CCD_1 * PC_1 + CCD_0 * PC_0)
            P_1 = (CCD_1 * PC_1) / (CCD_1 * PC_1 + CCD_0 * PC_0)
            if (P_0 > P_1):
                k.append(0)
            else:
                k.append(1)

        accu_score = accuracy_score(X_label_test.to_numpy(), np.array(k)) * 100
        D["Unimodal Gaussian"] = round(accu_score, 3)
        st.write(f"Accuracy: {round(accu_score, 3)}%")

        cm = confusion_matrix(X_label_test.to_numpy(), np.array(k))
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots()
        sn.heatmap(cm, annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)

    # Summary of accuracies
        st.sidebar.write("### Accuracy Scores for all Models")
        st.sidebar.write(D)
