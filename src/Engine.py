import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton, QLabel
from PyQt5.QtCore import Qt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, linear_model, naive_bayes, metrics
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC

#Load Training Model
model = 0
tfidf_vect = 0 
encoder = 0 

def load_model():
    global model, tfidf_vect, encoder
    Data = pd.read_csv("C:\\Users\\maria\\OneDrive\\Desktop\\MMA Git\\Complaints-Classification-MMA-Project\\consumer_complaints.csv", 
                       encoding='latin-1', low_memory=False)
    Data = Data[['product', 'consumer_complaint_narrative']]
    Data = Data[pd.notnull(Data['consumer_complaint_narrative'])]

    Data['category_id'] = Data['product'].factorize()[0]

    train_x, valid_x, train_y, valid_y = train_test_split(Data['consumer_complaint_narrative'], 
                                                          Data['product'])

    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.transform(valid_y)

    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(Data['consumer_complaint_narrative'])
    xtrain_tfidf = tfidf_vect.transform(train_x)
    xvalid_tfidf = tfidf_vect.transform(valid_x)

    
    logreg_model = linear_model.LogisticRegression(max_iter=500)
    nb_model = naive_bayes.MultinomialNB()
    svc_model = LinearSVC(max_iter=500, dual=False)

    logreg_model.fit(xtrain_tfidf, train_y)
    nb_model.fit(xtrain_tfidf, train_y)
    svc_model.fit(xtrain_tfidf, train_y)

    # Create Voting Classifier 
    voting_model = VotingClassifier(estimators=[
        ('logreg', logreg_model), 
        ('naive_bayes', nb_model),
        ('linear_svc', svc_model)
    ], voting='hard')

    voting_model.fit(xtrain_tfidf, train_y)

    model = voting_model

def classify_complaint(complaint_text):
    tfidf_complaint = tfidf_vect.transform([complaint_text])
    prediction_num = model.predict(tfidf_complaint)
    prediction_name = encoder.inverse_transform(prediction_num)
    return prediction_name[0]

class ComplaintClassifierApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.text_entry = QTextEdit(self)
        self.text_entry.setPlaceholderText("Enter the complaint text here...")
        layout.addWidget(self.text_entry)

        classify_button = QPushButton("Classify", self)
        classify_button.clicked.connect(self.on_classify_button_click)
        layout.addWidget(classify_button)

        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

        self.setLayout(layout)
        self.setWindowTitle("Financial Complaints Classifier")
        self.setGeometry(300, 300, 400, 300)

    def on_classify_button_click(self):
        complaint_text = self.text_entry.toPlainText().strip()
        if complaint_text:
            category = classify_complaint(complaint_text)
            self.result_label.setText(f"Category: {category}")
        else:
            self.result_label.setText("Please, introduce your complaint.")

if __name__ == '__main__':
    load_model()

    app = QApplication(sys.argv)
    classifier_app = ComplaintClassifierApp()
    classifier_app.show()
    sys.exit(app.exec_())

