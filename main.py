# https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
# https://cloud.google.com/vision/automl/docs/base64?hl=ja
# https://cloud.google.com/vision/automl/docs/predict?hl=ja#automl-nl-example-python

def recognize(request):
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'image' not in request.files:
            with open("./htmls/error.html", "r") as f:
                resultHTML = f.read()
                resultHTML=resultHTML.format(reason="ファイルが取得できないため")
            return resultHTML

        # データの取り出し
        file = request.files['image']

        # ファイル名がなかった時の処理
        if file.filename == '':
            with open("./htmls/error.html", "r") as f:
                resultHTML = f.read()
                resultHTML = resultHTML.format(reason="ファイル名が取得できないため")
            return resultHTML
        # ファイルのチェック
        if file:
            # and allwed_file(file.filename)
            import base64
            import cv2
            import numpy as np
            import Params
            import json
            import requests

            from google.cloud import automl_v1beta1
            from google.cloud.automl_v1beta1.proto import service_pb2
            from google.cloud.automl_v1beta1.gapic import enums


            # ペイロードの作成。 実はBase64にエンコードしておかないといけないらしい
            img_array = np.asarray(bytearray(file.stream.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, 1)
            img= cv2.resize(img,(640,480))
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            result, encimg = cv2.imencode(".jpg",img, encode_param)
            imageString = base64.b64encode(bytearray(encimg))
            imageString=imageString.decode()

            payload =  {
                "image": {
                    "imageBytes": imageString
                }
            }

            client = automl_v1beta1.AutoMlClient.from_service_account_json('key-file')
            prediction_client = automl_v1beta1.PredictionServiceClient.from_service_account_json('key-file')


            url="https://automl.googleapis.com/v1beta1/projects/479232824532/locations/{region}/models/{model}:predict"
            url=url.format(region=Params.compute_region,model=Params.model_id)
            headers = {"Authorization": "Bearer " + Params.AuthToken,"Content-Type": "application/json"}
            response=requests.post(url, data=json.dumps(payload),headers=headers)

            with open("./htmls/resultTrue.html", "r") as f:
                # 画像を含んだ結果をHTMLに埋め込む
                resultHTML = f.read()
                resultHTML=resultHTML.format(image_string=imageString,class_name=str(response),score="aa")
                #response.classification.score
                return resultHTML

    #
    with open("./htmls/error.html", "r") as f:
        resultHTML = f.read()
        resultHTML = resultHTML.format(reason="想定されていないため")
        return resultHTML

def allwed_file(filename):
    # .があるかどうかのチェックと、拡張子の確認
    # OKなら１、だめなら0
    ALLOWED_EXTENSIONS = set(['jpg','jpeg'])
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS