from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
from keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('model.keras')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    image = request.files['image']
    img = Image.open(image)
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Make a prediction
    prediction = model.predict(img)
    class_label = convert_prediction_to_class_label(prediction)

    # Return the result as HTML
    return render_template('predict.html', user_image=image, fruit=class_label)

def convert_prediction_to_class_label(prediction):
    if prediction[0][0] > 0.5:
        return 'Fractured'
    else:
        return 'Not Fractured'

if __name__ == '__main__':
    app.run(debug=True)