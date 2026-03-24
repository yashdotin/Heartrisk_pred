Heartrisk_pred

A simple Flask app that predicts heart disease risk using a trained machine learning model.

---

What it does

- Takes basic health inputs
- Runs them through a trained model
- Returns a risk prediction (High / Low)

---

Tech

- Python
- Flask
- Scikit-learn
- Pandas

---

Structure

Heartrisk_pred/
├── app/
│   ├── app.py
│   ├── train.py
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── templates/
│   └── static/
├── requirements.txt
└── README.md

---

Run locally

git clone https://github.com/yashdotin/Heartrisk_pred.git
cd Heartrisk_pred

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt

cd app
python app.py

---

Train model

cd app
python train.py

---

Note

This project is for learning/demo purposes.
Not intended for real medical use.

---

Author

Yash
