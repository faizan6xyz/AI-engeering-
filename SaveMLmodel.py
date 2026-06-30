import pickle
from sklearn.linear_model import LogisticRegression

# 1. Train Model
x = [[1], [2], [3]]  # Added more data for a better example
y = [1, 0, 1]
model = LogisticRegression()
model.fit(x, y)

# 2. Save Model (Corrected)
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)  # Added 'model' as the first argument

print("Model saved successfully.")

# 3. Load Model (Corrected)
with open("model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# 4. Test the loaded model
prediction = loaded_model.predict([[4]])
print(f"Prediction for input [4]: {prediction}")