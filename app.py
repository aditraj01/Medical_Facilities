import nltk
from flask import Flask, request, render_template, url_for,jsonify
import json
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import pickle
from joblib import dump,load
import random
app = Flask(__name__, static_url_path='/static')
with open('modelhealth.joblib', 'rb') as file:
    model1 = load(file)
with open('scalerhealth.joblib', 'rb') as file:
    scaler = load(file)
with open('modelmental.joblib', 'rb') as file:
    modelmental = load(file)
with open('encodermental.joblib', 'rb') as file:
    encoder = load(file)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['DEBUG'] = True

model = load_model('model.h5')
lemma = WordNetLemmatizer()
intents = json.load(open('intents.json'))
words = pickle.load(open('word.pkl', 'rb'))
classes = pickle.load(open('class.pkl', 'rb'))
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemma.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    cltn = np.zeros(len(words), dtype=np.float32)
    for word in sentence_words:
        for i, w in enumerate(words):
            if w == word:
                cltn[i] = 1
                if show_details:
                    print(f"Found '{w}' in bag")
    return cltn

def predict_class(sentence, model):
    l = bow(sentence, words, show_details=False)
    res = model.predict(np.array([l]))[0]

    ERROR_THRESHOLD = 0.25
    results = [(i, j) for i, j in enumerate(res) if j > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[k[0]], "probability": str(k[1])} for k in results]
    return return_list


'''def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
'''
def getResponse(ints, intents_json):
    try:
        tag = ints[0]['intent']
        for i in intents_json['intents']:
            if i['tag'] == tag:
                response = random.choice(i['responses'])
                treatment = random.choice(i['treatment']) if 'treatment' in i else ""
                home_remedies = random.choice(i['home_remedies']) if 'home_remedies' in i else ""
                response = response + treatment+home_remedies
                return response
    except IndexError:
        return ("I'm sorry, I didn't get that. Please try again.")

def chatbotResponse(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/bmi_health')
def bmi_health():
    return render_template('health.html')
@app.route('/bmi_health', methods=['POST','GET'])
def find_bmi():
    try:
        weight = float(request.form.get('weight'))
        height = float(request.form.get('height'))
        bmi = weight/((height/100)**2)
        if bmi<=18.5:
            return render_template('health.html',   BMI =round(bmi,2), Comment="Underweight")
        elif bmi>18.5 and bmi<=25.0:
            return render_template('health.html',   BMI =round(bmi,2), Comment="Normal")
        else:
            return render_template('health.html',  BMI =round(bmi,2), Comment="Overweight")
    except ValueError:
        return render_template('health.html', error='Invalid input. Please enter numeric values.'), 400



@app.route('/health')
def health():
    return render_template('health.html')


@app.route('/health', methods=['POST', 'GET'])
def index_predict():
    try:
        features = [
            request.form.get('feature1'), request.form.get('feature2'),
            request.form.get('feature3'), request.form.get('feature4'),
            request.form.get('feature5'), request.form.get('feature6'),
            request.form.get('feature7'), request.form.get('feature8'),
            request.form.get('feature9'), request.form.get('feature10'),
            request.form.get('feature11'), request.form.get('feature12')
        ]
        if '' in features:
            return render_template('health.html', error='Please fill in all fields.')

        features = [float(x) for x in features]
    except ValueError:
        return render_template('health.html', error='Invalid input. Please enter numeric values.')

    input_data = np.array([features])

    try:
        prediction = model1.predict(scaler.transform(input_data))  # Assuming model1 and scaler are initialized
        predicted_value = int(prediction[0])  # Get predicted result
    except Exception as e:
        return render_template('health.html', error=f'Error during prediction: {str(e)}')

    if predicted_value == 0:
        print(predicted_value)
        return render_template('result1.html', prediction1="Yes")
    else:
        print(predicted_value)
        return render_template('result1.html', prediction1="No")

@app.route('/mental')
def mental():
    return render_template('mental.html')
@app.route('/mental', methods=['POST','GET'])
def new_predict():
    try:
        features = [request.form.get('feature0'), request.form.get('feature1'), request.form.get('feature2'),
                    request.form.get('feature3'), request.form.get('feature5'),
                    request.form.get('feature6'), request.form.get('feature7'), request.form.get('feature8'),
                    request.form.get('feature9'), request.form.get('feature10'), request.form.get('feature11'),
                    request.form.get('feature12'), request.form.get('feature13'), request.form.get('feature14'),
                    request.form.get('feature15'), request.form.get('feature16'), request.form.get('feature17'),
                    request.form.get('feature18'), request.form.get('feature19'), request.form.get('feature20'),
                    request.form.get('feature21'), request.form.get('feature22')]
        if '' in features:
            return render_template('mental.html', error='Please fill in all fields.')
        print(features)
        features = [int(x) for x in features]
    except ValueError:
        return render_template('mental.html', error='Invalid input. Please enter numeric values.')
    input_data = np.array([features])
    print(input_data)
    prediction = modelmental.predict(input_data)
    if prediction==1:
        return render_template('result2.html', predictionnew="No")
    else:
        return render_template('result2.html', predictionnew="Yes")
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    response = chatbotResponse(msg)
    return response


if __name__ == '__main__':
    app.run()
