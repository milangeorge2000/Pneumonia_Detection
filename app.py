from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Define a flask app
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')








@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        import os, shutil
        folder = r'uploads'
        for filename in os.listdir(folder):
          file_path = os.path.join(folder, filename)
          if os.path.isfile(file_path) or os.path.islink(file_path):
              os.unlink(file_path)
          elif os.path.isdir(file_path):
             shutil.rmtree(file_path)
           
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        



        img = image.load_img(file_path,target_size=(224,224)) ##loading the image
        img = np.asarray(img) ##converting to an array
        img = img / 255 ##scaling by doing a division of 255
        img = np.expand_dims(img, axis=0) ##expanding the dimensions
        saved_model = load_model("modelvgg19_lung.h5") ##loading the model
        output = saved_model.predict(img)
        if output[0][0] > 0.5:
          result = "Pneumonia"
        else:
           result = "Normal"


        return result

if __name__ == '__main__':
    app.run(debug=True)