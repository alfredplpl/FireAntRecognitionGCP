# 参考URL
# https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
# https://cloud.google.com/vision/automl/docs/base64?hl=ja
# https://cloud.google.com/vision/automl/docs/predict?hl=ja#automl-nl-example-python
# https://qiita.com/iss-f/items/fcc766fca27f3685025d

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

        # ファイルの存在チェック
        if file:
            import base64
            import cv2
            import numpy as np
            import traceback

            import Params

            from google.cloud import automl_v1beta1

            try:
                # ペイロードの作成。 実はBase64にエンコードしておかないといけないらしい
                img_array = np.asarray(bytearray(file.stream.read()), dtype=np.uint8)
                img = cv2.imdecode(img_array, 1)
                img= cv2.resize(img,(640,480))
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                result, encimg = cv2.imencode(".jpeg",img, encode_param)
                imageBin = base64.b64encode(bytes(encimg))
                imageString=imageBin.decode()

                payload = {'image': {'image_bytes': bytes(encimg)}}
                client = automl_v1beta1.AutoMlClient.from_service_account_json('projectkey.json')
                prediction_client = automl_v1beta1.PredictionServiceClient.from_service_account_json('projectkey.json')

                params = {"score_threshold": bytes(b'0.5')}
                model_full_id = client.model_path(Params.project_id, Params.compute_region, Params.model_id)
                response = prediction_client.predict(model_full_id, payload,params)
                response=response.payload[0]

                with open("./htmls/result.html", "r") as f:
                    # 画像を含んだ結果をHTMLに埋め込む
                    resultHTML = f.read()
                    resultString="ある" if response.display_name=="fire_ant" else "ない"
                    resultHTML = resultHTML.format(image_string=imageString, class_name=response.display_name,
                                                   score=response.classification.score,result=resultString)
                    # response.classification.score
                    return resultHTML

            except:
                with open("./htmls/result.html", "r") as f:
                    # 画像を含んだ結果をHTMLに埋め込む
                    resultHTML = f.read()
                    resultHTML = resultHTML.format(image_string=imageString, class_name=traceback.format_exc(), score="aa",result="ある")
                    # response.classification.score
                    return resultHTML

    # GETなどの例外処理
    with open("./htmls/error.html", "r") as f:
        resultHTML = f.read()
        resultHTML = resultHTML.format(reason="想定されていないため")
        return resultHTML
