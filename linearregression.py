import pickle

model = pickle.load(open('linear.pkl', 'rb'))

pred = model.predict([[0.91, 0.43, 0.71, 0.59]])

print(pred[0,0])