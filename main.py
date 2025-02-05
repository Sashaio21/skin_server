from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Оставляет только ошибки

# Загрузка модели
MODEL_PATH = './assets/model/skin_disease_model_final.keras'
model = tf.keras.models.load_model(MODEL_PATH)

class_labels = {
    'akiec': 'Актинический кератоз',
    'bcc': 'Базальноклеточная карцинома',
    'bkl': 'Доброкачественный кератоз',
    'df': 'Дерматофиброма',
    'mel': 'Меланома',
    'nv': 'Меланоцитарный невус',
    'vasc': 'Сосудистые поражения'
}

app = Flask(__name__)
CORS(app) 

def predict_skin_disease(img, model):
    img = img.convert("RGB").resize((224, 224))  
    img_array = image.img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    prediction = model.predict(img_array)[0]  
    return prediction

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    img = Image.open(io.BytesIO(file.read()))
    predictions = predict_skin_disease(img, model)
    
    results = {class_labels[key]: f'{pred * 100:.2f}%' for key, pred in zip(class_labels.keys(), predictions)}
    predicted_class_key = list(class_labels.keys())[np.argmax(predictions)]
    predicted_class = class_labels[predicted_class_key]
    confidence = np.max(predictions) * 100
    
    return jsonify({
        'predicted_class': predicted_class,
        'confidence': f'{confidence:.2f}%',
        'all_predictions': results,
        'explanation': get_explanation(predicted_class_key),
    })

def get_explanation(predicted_class_key):
    explanations = {
        'akiec': "Актинический кератоз — это предраковое состояние кожи, вызванное солнцем.",
        'bcc': "Базальноклеточная карцинома — распространенный тип рака кожи.",
        'bkl': "Доброкачественный кератоз — неопасные новообразования.",
        'df': "Дерматофиброма — доброкачественная опухоль кожи.",
        'mel': "Меланома — злокачественная опухоль кожи, требующая внимания.",
        'nv': "Меланоцитарный невус — обычная родинка.",
        'vasc': "Сосудистые поражения — изменения, связанные с сосудами."
    }
    return explanations.get(predicted_class_key, "Неизвестный тип заболевания")

if __name__ == '__main__':
    app.run(debug=True)
