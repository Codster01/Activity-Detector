from ultralytics import YOLO
import argparse
import io
import os
from PIL import Image
import datetime
import pandas as pd
import torch
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"
list_dict = []
class_list = ['Instagram', 'None', 'Book']
bounding_boxes = pd.DataFrame()
@app.route("/predictor", methods=["GET", "POST"])
def predict():

    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model(img)
        res_plotted = results[0].plot(show_conf=True)
        for result in results:
            boxes = result.boxes
            masks = result.masks
            probs = result.probs 
        p=probs
        a = results[0].boxes.data
        a_list = [item.tolist() for item in a]
        
        px = pd.DataFrame(a).astype("float")
        bounding_boxes = px
        c = ""
        for _, row in px.iterrows():
            d = int(row[5])
            c = class_list[d]
            print(c)
        
        if not c:
            c = "Not Detected"

        now_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
        img_savename = f"static/{now_time}.png"

        print("Printing!!!!") # Update the global probs variable
        # print(results[0].boxes.data) # Update the global probs variable
        Image.fromarray(res_plotted).save(img_savename)
        

        return render_template("index.html", probs=c,image=img_savename) 
    return render_template("index.html") 

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/contactme" ,methods=["GET", "POST"])
def contactme():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        text= request.form.get("textarea")
        try:
            with open("contact_data.csv", "a") as file:
                file.write(f"Name: {name} , Email: {email} , Message: {text}\n")
        except Exception as e:
            print(f"Error writing to file: {e}")


        print("Name:", name)
        print("Email:", email)
        print("Text:", text)
    
        return "Response Recorded!"

    return render_template("contactme.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov8 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = YOLO('D:/Flask Project/best3.pt')  # force_reload = recache latest code
    
    app.run(host="0.0.0.0", port=args.port,debug=True)  # debug=True causes Restarting with stat
