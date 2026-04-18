# 💳 Credit Risk Prediction API

A production-grade Credit Risk Prediction system using XGBoost and FastAPI — with system design principles including Load Balancing and Redis Caching.

🔴 **Live Demo**: [huggingface.co/spaces/nitz0219/credit-risk-api](https://huggingface.co/spaces/nitz0219/credit-risk-api)

---

## 🧠 What It Does

Input loan applicant details → Get instant prediction on whether the applicant will **default or not default** on their loan.

---

## 🏗️ Architecture

```
User Input (Loan Details)
        ↓
   FastAPI REST API
        ↓
  Redis Cache Check
  ┌────────────────┐
  │ Cache HIT?     │ → Return instantly
  │ Cache MISS?    │ → Run XGBoost Model
  └────────────────┘
        ↓
  XGBoost Prediction
        ↓
  Cache Result in Redis
        ↓
  Return Risk Score + Decision
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | XGBoost |
| Backend | FastAPI |
| Caching | Redis |
| Frontend | Streamlit |
| Deployment | Docker + Hugging Face Spaces |
| Experiment Tracking | Weights & Biases |

---

## 🚀 System Design Features

- **Redis Caching** — same applicant profile returns cached result instantly
- **Load Balancer** — handles multiple concurrent prediction requests
- **Docker** — fully containerized deployment
- **REST API** — production-ready endpoints

---

## 📊 Model Performance

- Algorithm: XGBoost Classifier
- Features: Income, loan amount, credit history, employment status, etc.
- Evaluation: Cross-validation with grid search optimization

---

## 👨‍💻 Author

**Nitesh Nankani** — AI/ML Engineer  
[HuggingFace](https://huggingface.co/nitz0219) | [GitHub](https://github.com/niteshnankani-svg) | [LinkedIn](https://linkedin.com/in/niteshnankani)
