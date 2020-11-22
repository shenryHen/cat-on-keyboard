from flask import Flask, request, render_template
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras import utils
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

new_model = tf.keras.models.load_model('terribleCatModel.h5')
inputs=['a a s s e e r r o s o l']
# Check its architecture
new_model.summary()

app = Flask(__name__)

@app.route('/cat', methods=['GET', 'POST'])
def cat():
	if request.method == 'GET':
		return render_template(' base.html')
	if request.method == 'POST':
		print(request.form['input'])
		return 'cat'
		#print(request.get_json())

@app.route('/')
def index():
	return render_template('base.html')