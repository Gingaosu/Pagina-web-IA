from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from modelo import predict_image 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No se encontró el archivo"
    file = request.files['image']
    if file.filename == '':
        return "No se seleccionó ningún archivo"
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        predicted_class = predict_image(filepath)
        return jsonify({"prediction": predicted_class})
    
    return "Hubo un error al subir la imagen"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
