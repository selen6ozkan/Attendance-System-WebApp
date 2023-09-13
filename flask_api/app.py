import asyncio
import websockets
import base64
from cnn_model import load_cnn_model
from PIL import Image
import numpy as np
import io
from flask import Flask, request, jsonify

app = Flask(__name__)

# Flask API endpoint'ı
@app.route('/home', methods=['POST'])
def handle_post_request():
    data = request.get_json()
    image_data = data.get('image')

    # Load the CNN model
    cnn_model = load_cnn_model()

    # Perform face recognition using the loaded CNN model
    predicted_class = recognize_person(image_data, cnn_model)

    return jsonify({'predicted_class': predicted_class, 'success': True, 'message': 'Yüz tanıma işlemi başarıyla gerçekleştirildi.'})


# Yüz tanıma fonksiyonu
def recognize_person(data, model):
    # Burada yüz tanıma işlemini yapın ve sonucu döndürün
    return "person_1"  # Örnek olarak "person_1" döndürüyoruz, burada gerçek yüz tanıma işlemini yapmanız gerekecek


async def recognize_face(websocket, path):
    # Websocket bağlantısı başlatıldığında yapılacak işlemler
    async for message in websocket:
        # Gelen veriyi işle
        data = message["data"]
        # Yüz tanıma işlemlerini gerçekleştir ve sonucu al
        recognized_person_id = recognize_person(data)
        recognized_person_name = known_people.get(recognized_person_id, 'Unknown')  # Tanınan kişinin adını belirle
        # Tanınan kişinin adını Angular web sayfasına gönder
        response = {"person_name": recognized_person_name}
        await websocket.send(response)


start_server = websockets.serve(recognize_face, "localhost", 8765)

if __name__ == "__main__":
    # Flask API'yi başlatın ve belirtilen host ve portta dinleyin
    app.run(host="localhost", port=8765, debug=False)

    # Websocket'i başlatın
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
