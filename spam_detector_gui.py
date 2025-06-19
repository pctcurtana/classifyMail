import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import joblib
import threading


class SpamDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kiểm Tra Email Spam")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Load model và vectorizer
        try:
            self.model = joblib.load('spam_model.pkl')
            self.vectorizer = joblib.load('vectorizer.pkl')
            self.model_loaded = True
        except FileNotFoundError:
            self.model_loaded = False
            messagebox.showerror("Lỗi", "Không tìm thấy model! Vui lòng chạy train_model.py trước.")
        
        self.setup_ui()
        
    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=70)
        header_frame.pack(fill='x', padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame, 
            text="EMAIL SPAM DETECTOR", 
            font=('Arial', 18, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Input section
        input_label = tk.Label(
            main_frame,
            text="📧 Nhập nội dung email để kiểm tra:",
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        input_label.pack(anchor='w', pady=(0, 10))
        
        # Text input area
        self.text_input = scrolledtext.ScrolledText(
            main_frame,
            height=8,
            width=70,
            font=('Arial', 11),
            wrap=tk.WORD,
            relief='ridge',
            bd=2
        )
        self.text_input.pack(fill='both', expand=True, pady=(0, 20))
        
        # Button frame
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(fill='x', pady=(0, 20))
        
        # Analyze button
        self.analyze_btn = tk.Button(
            button_frame,
            text="🔍 PHÂN TÍCH",
            font=('Arial', 12, 'bold'),
            bg='#3498db',
            fg='white',
            height=2,
            width=15,
            relief='flat',
            cursor='hand2',
            command=self.analyze_email
        )
        self.analyze_btn.pack(side='left', padx=(0, 10))
        
        # Clear button
        clear_btn = tk.Button(
            button_frame,
            text="🗑️ XÓA",
            font=('Arial', 12, 'bold'),
            bg='#95a5a6',
            fg='white',
            height=2,
            width=10,
            relief='flat',
            cursor='hand2',
            command=self.clear_text
        )
        clear_btn.pack(side='left', padx=(0, 10))
        
        # Sample button
        sample_btn = tk.Button(
            button_frame,
            text="📝 MẪU SPAM",
            font=('Arial', 12, 'bold'),
            bg='#e74c3c',
            fg='white',
            height=2,
            width=12,
            relief='flat',
            cursor='hand2',
            command=self.load_sample_spam
        )
        sample_btn.pack(side='left')
        
        # Result section
        result_label = tk.Label(
            main_frame,
            text="🔍 Kết quả phân tích:",
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        result_label.pack(anchor='w', pady=(20, 10))
        
        # Result display frame
        self.result_frame = tk.Frame(main_frame, bg='#ffffff', relief='ridge', bd=2)
        self.result_frame.pack(fill='x', pady=(0, 20))
        
        self.result_text = tk.Label(
            self.result_frame,
            text="📋 Chưa có kết quả phân tích",
            font=('Arial', 14),
            bg='#ffffff',
            fg='#7f8c8d',
            height=3,
            justify='center'
        )
        self.result_text.pack(fill='x', padx=20, pady=20)
        
        # Status bar
        self.status_label = tk.Label(
            self.root,
            text="✅ Sẵn sàng phân tích" if self.model_loaded else "❌ Model chưa được tải",
            font=('Arial', 10),
            bg='#ecf0f1',
            fg='#2c3e50',
            relief='sunken',
            bd=1
        )
        self.status_label.pack(side='bottom', fill='x')
        
    def analyze_email(self):
        if not self.model_loaded:
            messagebox.showerror("Lỗi", "Model chưa được tải! Vui lòng chạy train_model.py trước.")
            return
            
        email_content = self.text_input.get("1.0", tk.END).strip()
        if not email_content:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập nội dung email!")
            return
        
        # Disable button during analysis
        self.analyze_btn.config(state='disabled', text="⏳ ĐANG PHÂN TÍCH...")
        self.status_label.config(text="⏳ Đang phân tích email...")
        
        # Run analysis in separate thread to prevent GUI freezing
        threading.Thread(target=self.perform_analysis, args=(email_content,), daemon=True).start()
    
    def perform_analysis(self, email_content):
        try:
            # Vectorize input
            input_vect = self.vectorizer.transform([email_content])
            
            # Make prediction
            prediction = self.model.predict(input_vect)[0]
            probability = self.model.predict_proba(input_vect)[0]
            
            # Get confidence
            confidence = max(probability) * 100
            
            # Update UI in main thread
            self.root.after(1000, self.update_result, prediction, confidence, email_content)
            
        except Exception as e:
            self.root.after(0, self.show_error, str(e))
    
    def update_result(self, prediction, confidence, email_content):
        # Re-enable button
        self.analyze_btn.config(state='normal', text="🔍 PHÂN TÍCH")
        
        if prediction == 1:
            result_text = f"🚨 SPAM EMAIL\n⚠️ Độ tin cậy: {confidence:.1f}%"
            bg_color = '#e74c3c'
            fg_color = 'white'
        else:
            result_text = f"✅ EMAIL AN TOÀN\n✨ Độ tin cậy: {confidence:.1f}%"
            bg_color = '#27ae60'
            fg_color = 'white'
        
        # Update result display
        self.result_text.config(
            text=result_text,
            bg=bg_color,
            fg=fg_color,
            font=('Arial', 14, 'bold')
        )
        self.result_frame.config(bg=bg_color)
        
        self.status_label.config(text="✅ Phân tích hoàn tất")
    
    def show_error(self, error_msg):
        self.analyze_btn.config(state='normal', text="🔍 PHÂN TÍCH")
        self.status_label.config(text="❌ Có lỗi xảy ra")
        messagebox.showerror("Lỗi", f"Có lỗi xảy ra: {error_msg}")
    
    def clear_text(self):
        self.text_input.delete("1.0", tk.END)
        self.result_text.config(
            text="📋 Chưa có kết quả phân tích",
            bg='#ffffff',
            fg='#7f8c8d',
            font=('Arial', 14)
        )
        self.result_frame.config(bg='#ffffff')
        self.status_label.config(text="✅ Sẵn sàng phân tích")
    
    def load_sample_spam(self):
        sample_spam = """Congratulations! You have won $1000000 in our lottery! 
Click here now to claim your prize: http://fake-lottery.com
Send your bank details immediately to claim your money!
This offer expires in 24 hours! Act now!
Call +1-800-FAKE-NUM for more details."""
        
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", sample_spam)

def main():
    root = tk.Tk()
    app = SpamDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 