from flask import Flask, render_template, request, jsonify
from PIL import Image
from io import BytesIO
from keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np
import os

labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc','mel']
dict = {
  'akiec': 'Dày sừng ánh sáng',
  'bcc': 'Ung thư biểu mô tế bào đáy',
  'bkl': 'Chứng dày sừng tiết bã',
  'df' : 'U sợi bì lành tính',
  'nv' : 'Nốt ruồi',
  'vasc' : 'Ung thư hắc tố da',
  'mel' : 'Thương tổn mạch máu',
}

app = Flask(__name__)

UPLOAD_FOLDER = 'static/images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def analyzer():
    if request.method == "POST":
        image = request.files['image']
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image = Image.open(image_path).resize((100, 100))
        image = (np.asarray(image) - np.mean(image)) / np.std(image)

        model_path = os.path.join("models", "model.h5")
        model = load_model(model_path)

        predictions = model.predict(np.expand_dims(image, axis=0))
        predicted_class = np.argmax(predictions)
        output = dict[labels[predicted_class]]

        probabilities = tf.keras.backend.softmax(predictions).numpy()
        probabilities_dict  = {}
        probabilities_dict['Dày sừng ánh sáng'] = probabilities[0][0]
        probabilities_dict['Ung thư biểu mô tế bào đáy'] = probabilities[0][1]
        probabilities_dict['Chứng dày sừng tiết bã'] = probabilities[0][2]
        probabilities_dict['U sợi bì lành tính'] = probabilities[0][3]
        probabilities_dict['Nốt ruồi'] = probabilities[0][4]
        probabilities_dict['Ung thư hắc tố da'] = probabilities[0][5]
        probabilities_dict['Thương tổn mạch máu'] = probabilities[0][6]

        return render_template("index.html", image_path = image_path.replace('static/', ''),
                               output = output, dict = probabilities_dict)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080)
