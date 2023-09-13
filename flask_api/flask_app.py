import importlib
from flask import Flask, request, jsonify,render_template
import base64
from io import BytesIO
import cv2
import numpy as np

app = Flask(__name__)

def load_cnn_model():
    return importlib.import_module('cnn_model').load_cnn_model()

def preprocess_image(image_data):
    decoded_data = base64.b64decode(image_data.split(',')[1])
    image_np = np.frombuffer(decoded_data, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image

def recognize_person(image_data, model):
    image = preprocess_image(image_data)
    # Burada yüz tanıma işlemini gerçekleştiren kodları çağırın
    # Örneğin, model.recognize_person(image) şeklinde modeli kullanabilirsiniz
    # Daha sonra yüz tanıma sonucunu alarak predicted_class değişkenine atayın
    predicted_class = "Tanınan Kişi"
    return predicted_class

@app.route('/home', methods=['POST'])
def index():
    return render_template("home.component.html")

def handle_post_request():
    data = request.get_json()
    image_data = data.get('image')

    # Load the CNN model
    cnn_model = load_cnn_model()

    # Perform face recognition using the loaded CNN model
    predicted_class = recognize_person(image_data, cnn_model)

    return jsonify({'predicted_class': predicted_class, 'success': True, 'message': 'Yüz tanıma işlemi başarıyla gerçekleştirildi.'})


if __name__ == "__main__":
    app.run(host="localhost", port=8765, debug=False) 
