# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Đọc dữ liệu
data = pd.read_csv("sms.tsv", sep='\t', header=None, names=['label', 'message'])
# source data: https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv
data['label_num'] = data.label.map({'ham': 0, 'spam': 1})

# 2. Tách train/test
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label_num'], test_size=0.2, random_state=42
)

# 3. Vector hóa bằng Tfidf
vectorizer = TfidfVectorizer(
    stop_words='english'
)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# 4. Huấn luyện mô hình
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# 5. Đánh giá
y_pred = model.predict(X_test_vect)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Số dòng dữ liệu huấn luyện:", len(data))

# 6. Lưu mô hình và vectorizer
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("\nĐã lưu model và vectorizer thành công!\n")
