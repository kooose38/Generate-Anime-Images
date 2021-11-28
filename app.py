from flask import Flask, render_template, request
import uuid 

from generated import generated_img
from detect import detect 

app = Flask(__name__, static_folder='./templates/images') # 静的な画像を読み込むための宣言

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
   uid = str(uuid.uuid4())[:4]
   img = generated_img(uid)
   print("[INFO] successfully GAN.")

   detected_predict = detect(img)

   print("[INFO] successfully DETECT.")

   result_output = {
      "path": f"images/{uid}.png", 
      "predict": detected_predict
   }

   return render_template("result.html", result_output=result_output)


if __name__ == "__main__":
   app.run(debug=True, port=8080)