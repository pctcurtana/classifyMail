# predict_spam.py

import joblib

# Load model và vectorizer đã huấn luyện
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Nhập nội dung từ bàn phím để dự đoán
while True:
    user_input = input("\n📧 Nhập nội dung email để kiểm tra (hoặc gõ 'exit' để thoát):\n> ")
    if user_input.strip().lower() == 'exit':
        print("👋 Thoát chương trình.")
        break

    input_vect = vectorizer.transform([user_input])
    prediction = model.predict(input_vect)[0]

    label = "SPAM ❌" if prediction == 1 else "NON-SPAM ✅"
    print(f"🔍 Dự đoán: {label}")
