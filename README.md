[README.md](https://github.com/user-attachments/files/27957474/README.md)
# OptiPay AI

## Smart Payment Recommendation System using Machine Learning

OptiPay AI is an ML-powered cashback recommendation system that predicts the best payment option for a transaction based on platform, category, bank offers, and payment methods.

The system analyzes transaction details and recommends the payment method that provides the highest cashback and savings.

---

## Project Objective

Online shopping platforms provide multiple bank offers and cashback options, making it difficult for users to identify the best payment method.

OptiPay AI solves this problem by:
- Comparing available offers
- Predicting cashback using Machine Learning
- Recommending the best payment option instantly

---

## Features

- ML-based cashback prediction
- Smart payment recommendation engine
- Interactive Streamlit web application
- Offer comparison table
- Cashback and savings percentage calculation
- Multiple categories and platforms supported

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- TF-IDF Vectorizer

---

## Machine Learning Models Used

### Random Forest Regressor
- MAE: ₹0.63
- RMSE: ₹5.81
- R² Score: 0.9992

### Gradient Boosting Regressor
- MAE: ₹1.30
- RMSE: ₹3.51
- R² Score: 0.9997

Final Selected Model:
- Gradient Boosting Regressor

---

## Dataset Information

The dataset used in this project is a synthetic dataset created to simulate real-world Indian cashback transactions.

### Dataset Generation Logic
- Realistic platforms, banks, and payment methods were defined
- Transaction combinations were generated randomly
- Business rules were applied:
  - Cashback applicable only if minimum transaction amount is satisfied
  - Cashback capped at maximum cashback limit
  - Effective savings percentage calculated

Total Rows:
- Approximately 3000 transactions

---

## Project Workflow

1. User enters transaction amount
2. User selects category and platform
3. System fetches available offers
4. Features are generated for ML prediction
5. Model predicts cashback amount
6. Best payment option is recommended

---

## Categories Included

- Fashion
- Electronics
- Grocery
- Food Delivery
- Travel

---

## Streamlit Deployment

Run the application using:

```bash
streamlit run app.py
```

---

## Files Included

- `app.py` → Streamlit application
- `best_model.pkl` → Trained ML model
- `tfidf_vectorizer.pkl` → TF-IDF vectorizer
- `label_encoders.pkl` → Saved label encoders
- `OptiPay_AI.ipynb` → Model training notebook
- `requirements.txt` → Required libraries

---

## Future Enhancements

- Real-time bank offer integration
- User personalization
- Recommendation history
- More advanced ranking models
- API integration with payment platforms

---

## Author

Shruti Mandale
