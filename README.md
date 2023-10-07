# email-spam-detection
Email spam detection is the process of identifying and filtering out unwanted or unsolicited emails, commonly referred to as spam, from legitimate emails in your inbox. The primary goal of email spam detection is to improve the user's email experience by reducing the clutter and potential security risks associated with spam emails.
Content Filtering: Content-based filtering is one of the most common methods used to detect spam emails. It involves analyzing the content of an email message, including text, images, links, and attachments, to determine whether it is spam. This technique uses various algorithms and heuristics to identify common spam characteristics, such as specific keywords, phrases, or patterns.

Sender Reputation: Another effective approach is to evaluate the reputation of the sender. Email servers maintain databases of known spammers and reputable senders. If an email comes from an IP address or domain with a bad reputation, it is more likely to be classified as spam. Conversely, emails from trusted sources are more likely to reach the inbox.

Machine Learning: Machine learning techniques are increasingly used for spam detection. Algorithms like Naive Bayes, Support Vector Machines, and deep learning models can learn from historical data to classify emails as spam or not based on various features, including email content, sender information, and user behavior.

Blacklists and Whitelists: Blacklists contain known spammer email addresses, domains, or IP addresses, and emails from these sources are automatically marked as spam. Conversely, whitelists contain trusted senders and ensure their emails reach the inbox. These lists are maintained by email service providers and users can also customize them.

User Feedback: Some email providers allow users to mark emails as spam or not spam manually. This feedback helps improve the accuracy of spam filters over time. Machine learning algorithms can incorporate this information to make better predictions.

Header Analysis: The email header contains information about the sender, route, and other metadata. Analyzing the header can reveal anomalies or suspicious patterns that suggest spam.

Bayesian Filters: Bayesian filtering is a statistical approach that calculates the probability of an email being spam based on the occurrence of certain words or phrases. It assigns probabilities to different characteristics of emails and uses them to classify messages as spam or not.

Real-time Analysis: Spam detection systems often work in real-time, quickly assessing incoming emails as they arrive in the inbox. This helps in blocking spam before it reaches the user's attention.

Behavioral Analysis: Advanced spam detection systems also consider user behavior. For example, they may analyze how the recipient interacts with emails, including whether they open emails, click links, or mark messages as spam.

Adaptive Filters: Spam detection systems continually adapt and update their algorithms to stay ahead of evolving spam techniques. They incorporate new data and patterns to improve accuracy.

Effective email spam detection is crucial to ensuring the security, privacy, and usability of email services. By employing a combination of these techniques, email providers can significantly reduce the volume of spam that reaches users' inboxes, allowing for a more efficient and safe email communication experience.
the tools used for data aquisition,data pre processing starts from here
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('/content/sample_data/mail_data (2).csv')

print(df)

data = df.where((pd.notnull(df)),'')
data.head(10)

data.info()

data.shape

data.loc[data['Category'] == 'spam' , 'Category' ,] = 0
data.loc[data['Category'] == 'ham' , 'Category',] = 1


X = data['Message']
Y = data['Category']

print(X)

print(Y)

from pandas.core.common import RandomState
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=3)

print(X.shape)

print(X_train.shape)

print(X_test.shape)

print(Y.shape)
print(Y_train.shape)
print(Y_test.shape)

feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase= True)
X_train_features = feature_extraction.fit_transform(X_train) 
X_test_features = feature_extraction.transform(X_test)
Y_train = Y_train.astype('int')
Y_test=Y_test.astype('int')


print(X_train)

print(X_train_features)

model = LogisticRegression()
model.fit(X_train_features,Y_train)

prediction_on_training_data = model.predict(X_train_features) 
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

print('Acc on training data :', accuracy_on_training_data)

prediction_on_test_data = model.predict(X_test_features) 
accuracy_on_test_data = accuracy_score (Y_test, prediction_on_test_data)

print('Acc on test data :', accuracy_on_test_data)

#spam detection
input_your_mail = ['Thanks for your subscription to Ringtone UK your mobile will be charged Â£5/month Please confirm by replying YES or NO. If you reply NO you will not be charged']
input_data_features = feature_extraction.transform(input_your_mail)
prediction = model.predict(input_data_features)
print(prediction)
if(prediction[0]==1):
  print('ham_mail')
else:
  print('spam_mail')

input_your_mail = ['Oops, I ll let you know when my roommates done']
input_data_features = feature_extraction.transform(input_your_mail)
prediction = model.predict(input_data_features)
print(prediction)
if(prediction[0]==1):
  print('ham_mail')
else:
  print('spam_mail')




