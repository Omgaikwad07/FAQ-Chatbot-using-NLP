import json
import nltk
import string
import numpy as np
import tkinter as tk
from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data 
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ---------------- LOAD FAQ DATA ----------------

with open("faqs.json", "r") as f:
    faqs = json.load(f)

questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]

stop_words = set(stopwords.words('english'))

# ---------------- PREPROCESS FUNCTION ----------------

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [
        word for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]
    return " ".join(tokens)

processed_questions = [preprocess(q) for q in questions]

# ---------------- VECTORIZE FAQ QUESTIONS ----------------

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_questions)

# ---------------- CHATBOT LOGIC ----------------

def chatbot_response(user_input):
    user_input_processed = preprocess(user_input)
    user_vector = vectorizer.transform([user_input_processed])

    similarities = cosine_similarity(user_vector, faq_vectors)
    best_match_index = np.argmax(similarities)

    if similarities[0][best_match_index] < 0.2:
        return "Sorry, I couldn't understand your question."

    return answers[best_match_index]

# ---------------- TKINTER UI ----------------

def send_message():
    user_text = entry.get()
    if user_text.strip() == "":
        return

    chat_area.insert(tk.END, "You: " + user_text + "\n")
    entry.delete(0, tk.END)

    response = chatbot_response(user_text)
    chat_area.insert(tk.END, "Bot: " + response + "\n\n")

    chat_area.yview(tk.END)

def exit_app():
    root.destroy()

# Main window
root = tk.Tk()
root.title("FAQ Chatbot")
root.geometry("520x480")
root.configure(bg="#F4F6F8")

# Title label
title = tk.Label(
    root,
    text="FAQ Chatbot",
    font=("Segoe UI", 14, "bold"),
    bg="#F4F6F8",
    fg="#333333"
)
title.pack(pady=8)

# Chat display
chat_area = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD,
    width=60,
    height=18,
    bg="#FFFFFF",
    fg="#000000",
    font=("Segoe UI", 10),
    borderwidth=1,
    relief="solid"
)
chat_area.pack(padx=12, pady=8)
chat_area.insert(tk.END, "Bot: Hello! Ask me your questions.\n\n")

# Input frame
input_frame = tk.Frame(root, bg="#F4F6F8")
input_frame.pack(pady=8)

entry = tk.Entry(
    input_frame,
    width=38,
    font=("Segoe UI", 10),
    relief="solid",
    borderwidth=1
)
entry.pack(side=tk.LEFT, padx=6)

send_button = tk.Button(
    input_frame,
    text="Send",
    width=10,
    bg="#4A90E2",
    fg="white",
    font=("Segoe UI", 10, "bold"),
    relief="flat",
    command=send_message
)
send_button.pack(side=tk.LEFT)

# Exit button
exit_button = tk.Button(
    root,
    text="Exit",
    width=10,
    bg="#E74C3C",
    fg="white",
    font=("Segoe UI", 10, "bold"),
    relief="flat",
    command=exit_app
)
exit_button.pack(pady=6)

# Run app
root.mainloop()
