from flask import Flask, render_template,  request, jsonify 
import os
from prediction_service import prediction_ as predictor
from pprint import pprint

templatess = os.path.join('webapp/templates')
static = os.path.join('webapp/static')



app = Flask("__name__", template_folder=templatess, static_folder=static)

# @app.route('/')
# def index():
#     return "Hello World"

@app.route("/", methods=["GET", "POST"])
def index1():
    if request.method == "POST":
        try:
            if request.form:
                data_req = dict(request.form)
                # print(data_req)

                response = predictor.form_response(data_req)
                # if response == "error":

                
                return render_template("home.html", response=response)
            # elif request.json:
            #     response = prediction.api_response(request.json)
            #     return jsonify(response)
        except Exception as e:
            # print(e)
            # error ={"error": "Something went wrong try again"}
            error = {"error": e}
            return render_template("404.html", error=error)
    else:
        return render_template("home.html", response="")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5100, debug=True)


