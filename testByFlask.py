__author__ = "Alfred Increment"
__version__ = "0.0.1"
__license__ = "Apache License 2.0"

from flask import Flask, request

app = Flask(__name__)

# ファイルを受け取る方法の指定
@app.route('/',methods=["POST"])
def serve():
    return recognize(request.get_data())

def recognize(request):
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'image' not in request.files:
            with open("./htmls/error.html", "r") as f:
                resultHTML = f.read()
                resultHTML=resultHTML.format(reason="ファイルが取得できないため")
            return resultHTML

        # ファイルに関するデータの取り出し
        file = request.files['image']

        # ファイル名がなかった時の処理
        if file.filename == '':
            with open("./htmls/error.html", "r") as f:
                resultHTML = f.read()
                resultHTML = resultHTML.format(reason="ファイル名が取得できないため")
            return resultHTML

        # ファイルの存在チェック
        if file:
            import base64
            import cv2
            import numpy as np

            # 非公開なパラメータを入れておくところ
            import Params

            # ペイロードの作成。大きすぎる画像をリサイズ。
            img_array = np.asarray(bytearray(file.stream.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, 1)
            if(img.shape[1]>640):
                img= cv2.resize(img,(640,img.shape[0]*640/img.shape[1]))
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            result, encimg = cv2.imencode(".jpeg",img, encode_param)

            # 画像をはbase64にエンコードしておく
            imageBin = base64.b64encode(bytes(encimg))
            imageString=imageBin.decode()

            #ToDo: Predict class
            response=None

            with open("./htmls/result.html", "r") as f:
                # 画像を含んだ結果をHTMLに埋め込む
                # 画像を埋め込んだ理由はCloud FunctionsからStorageにアップロードできないように作られているためである
                # （不正なアップローダー防止の対策とはいえ、めんどくさい仕様だ・・・
                resultHTML = f.read()
                resultString="ある" if response.display_name=="fire_ant" else "ない"
                resultHTML = resultHTML.format(image_string=imageString, class_name=response.display_name,
                                               score=response.classification.score,result=resultString)

                return resultHTML

    # GETなどの例外処理
    with open("./htmls/error.html", "r") as f:
        resultHTML = f.read()
        resultHTML = resultHTML.format(reason="想定されていないため")
        return resultHTML

if __name__ == '__main__':
    app.run()

# References
# https://a2c.bitbucket.io/flask/
# https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
# https://cloud.google.com/vision/automl/docs/base64?hl=ja
# https://cloud.google.com/vision/automl/docs/predict?hl=ja#automl-nl-example-python
# https://qiita.com/iss-f/items/fcc766fca27f3685025d