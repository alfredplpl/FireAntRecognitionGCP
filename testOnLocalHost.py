__author__ = "Alfred Increment"
__version__ = "0.0.1"
__license__ = "Apache License 2.0"

from flask import Flask, request
import main

app = Flask(__name__)

# ファイルを受け取る方法の指定
@app.route('/',methods=["POST"])
def serve():
    return main.recognizeByAutoKeras(request)


if __name__ == '__main__':
    app.run('127.0.0.1', 8000, debug=True)

# References
# https://a2c.bitbucket.io/flask/
# https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
# https://cloud.google.com/vision/automl/docs/base64?hl=ja
# https://cloud.google.com/vision/automl/docs/predict?hl=ja#automl-nl-example-python
# https://qiita.com/iss-f/items/fcc766fca27f3685025d
# https://qiita.com/itkr/items/d3f8b4f8ff9becf101b4