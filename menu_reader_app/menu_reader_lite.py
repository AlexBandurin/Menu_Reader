#app.py
from flask import Flask, flash, request, redirect, url_for, render_template, session 
import os
# from werkzeug.utils import secure_filename
# import cv2
# import numpy as np
# import pandas as pd
import requests
import json
# import base64
import pickle
import numpy as np

app = Flask(__name__)

# def get_memory_usage():
#     process = psutil.Process(os.getpid())
#     return process.memory_info().rss / (1024 ** 2)  # Return in MB
 
app.secret_key = "secret key"

THUMBNAILS_FOLDER = 'static/thumbnails/'
UPLOAD_FOLDER = 'static/uploads/'

file_images = pickle.load(open('file_images.pkl', "rb")) 
file_dimensions = pickle.load(open('file_dimensions.pkl', "rb")) 

app.config['THUMBNAILS_FOLDER'] = THUMBNAILS_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# Assuming these are the images available for clicking
available_images = os.listdir(UPLOAD_FOLDER)

#thumbnails = os.listdir(app.config['THUMBNAILS_FOLDER'])


thumbnails = ['parkers_menu.jpeg', 'aubrees_menu.jpeg' ,'meatheads_menu.jpeg', 'portillos_menu.jpeg','roanoke_menu.jpeg']

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)
    
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_filename(file_path):
    parts = file_path.split('/', 1)
    if len(parts) > 1:
        return parts[1]
    return file_path
     
@app.route('/')
def home():

    return render_template('index.html', thumbnails=thumbnails)#, uploaded_images=uploaded_images)
 
# @app.route('/upload', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         flash('No file part')
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         flash('No image selected for uploading')
#         return redirect(request.url)
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#         # Add the uploaded filename to the session
#         session.pop('selected_filename', None)
#         session['selected_filename_path'] = UPLOAD_FOLDER + filename

#         return redirect(url_for('display_image', filename=filename))
#     else:
#         flash('Image type not allowed. Allowed image types are: png, jpg, and jpeg')
#         return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    if filename in thumbnails:
        session['image_source'] = 'thumbnails'
    else:
        session['image_source'] = 'uploads'

    source = session.pop('image_source', 'uploads')  # default to 'uploads' if not set and pop it

    print(source)
    if source == 'uploads':
        return render_template('index.html', filename='uploads/' + filename, thumbnails = thumbnails)
    else:
        session['image_source'] = 'thumbnails'
        session.pop('selected_filename', None)
        session['selected_filename_path'] = THUMBNAILS_FOLDER + filename
        return render_template('index.html', filename='thumbnails/' + filename, thumbnails = thumbnails)


@app.route('/display/menu_read', methods = ['GET','POST'])
def new_function():
    file_path = session.get('selected_filename_path')
    filename = extract_filename(file_path)
   
    # print("File path:", file_path)
    
    try: 
        # image0 = cv2.imread(file_path)
        # # Convert the image to RGB (OpenCV loads images in BGR format)
        # image = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)

        # # Convert the image array to a byte stream
        # _, buffer = cv2.imencode('.png', image)

        # # Convert the byte stream to a base64 string
        # image_base64 = base64.b64encode(buffer).decode('utf-8')

        # # Create a JSON payload
        # data = {
        #     "image": image_base64
        # }
        file_name = extract_filename(filename)
        # print('filename: ',file_name)
        result = file_images.get(file_name)
        height_img, width_img = file_dimensions.get(file_name)
        # print('height, width: ', height_img, width_img)

        data = {
            "result": result,
            "height_img": height_img,
            "width_img": width_img
        }
        
        json_data = json.dumps(data, cls=NumpyEncoder)

        function_app_url = "https://menu-reader.azurewebsites.net/api/MyFunction?code=4XMM6DmQmdPpC95hVvNjIQWggkOL5KKktzaagfCgT_IsAzFu1yvxUg=="

        headers = {'Content-Type': 'application/json'}
        response = requests.post(function_app_url, data=json_data, headers=headers)

        if response.status_code == 200:

            response_txt = response.text
            valid_json_string = response_txt.replace("'", "\"")

            categories = json.loads(valid_json_string)
            # print(categories)
            return render_template('index.html', categories=categories, thumbnails = thumbnails, filename = filename)
        else:
            # print('cannot connect')
            flash(f"Error: Cannot connect to function")
            return render_template('index.html', thumbnails = thumbnails, filename = filename)
                
    
    except Exception as e:
        flash(f"Error: {e}")
        print('Error: ', e)
        return render_template('index.html', thumbnails = thumbnails, filename = filename)
 
if __name__ == "__main__":
    app.run(debug = False)


