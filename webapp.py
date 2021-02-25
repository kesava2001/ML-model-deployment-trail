from flask import Flask, request, render_template
import numpy as np
import pickle
app = Flask(__name__, template_folder='html files')
model = pickle.load(open('linear.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('new form.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = np.round(prediction[0], 2)

    return render_template('new form.html', put_text='Predicted value is {}'.format(output[0]))

if __name__=="__main__":
    app.run(debug=True)