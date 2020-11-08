import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from PIL import Image

import os
from flask import Flask
from flask import request
from flask import render_template

app = Flask(__name__)
UPLOAD_FOLDER = "static"

def get_model():
	global model
	model = torchvision.models.densenet121(pretrained=True)
	num_ftrs = model.classifier.in_features
	model.classifier = nn.Sequential(
			nn.Linear(num_ftrs, 500),
			nn.Linear(500, 2)
		)
	model.load_state_dict(torch.load('ckpt_densenet121_catdog.pth', map_location=torch.device("cpu")))
	model.to("cpu")
	model.eval()
	print("model loaded")

def preprocess_image(image, target_size):
	if image.mode != "RGB":
		image = image.convert("RGB")
	transform = transforms.Compose([
			transforms.Resize(target_size),
			transforms.ToTensor()
		])
	image = transform(image).unsqueeze(dim=0)
	return image

def predict(image_path):
	image = Image.open(image_path)
	processed_image = preprocess_image(image, target_size=(128, 128))
	with torch.no_grad():
		get_model()
		output = model(processed_image)
		pred = torch.argmax(output, dim=1)
		res =  "dog" if pred.item() else "cat"
	return res

@app.route("/", methods=["GET", "POST"])
def upload_predict():
	if request.method == "POST":
		image_file = request.files["image"]
		if image_file:
			image_location = os.path.join(
					UPLOAD_FOLDER,
					image_file.filename
				)
			image_file.save(image_location)
			pred = predict(image_location)
			return render_template("index.html", prediction=pred, image_loc=image_file.filename)		
	return render_template("index.html", prediction=0, image_loc=None)

if __name__ == "__main__":
	print("loading pytorch model")
	get_model()
	app.run(debug=False)
