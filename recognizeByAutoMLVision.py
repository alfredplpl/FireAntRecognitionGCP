__author__ = "Alfred Increment"
__version__ = "0.0.1"
__license__ = "Apache License 2.0"

def recognizeByAutoMLVision(request):
    from flask import render_template

    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'image' not in request.files:
            return render_template("error.html",reason="ファイルが取得できないため")

        # ファイルに関するデータの取り出し
        file = request.files['image']

        # ファイル名がなかった時の処理
        if file.filename == '':
            return render_template("error.html",reason="ファイル名が取得できないため")

        # ファイルの存在チェック
        if file:
            import base64
            import cv2
            import numpy as np

            # 非公開なパラメータを入れておくところ
            import Params

            from google.cloud import automl_v1beta1

            MAX_WIDTH=640
            MAX_HEIGHT=480

            # ペイロードの作成。大きすぎる画像をリサイズ。
            img_array = np.asarray(bytearray(file.stream.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, 1)

            if(img.shape[0]>MAX_HEIGHT):
                img= cv2.resize(img,(int(img.shape[1]*MAX_WIDTH//img.shape[0]),MAX_HEIGHT))
            if(img.shape[1]>MAX_WIDTH):
                img= cv2.resize(img,(MAX_WIDTH,int(img.shape[0]*MAX_HEIGHT//img.shape[1])))

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            result, encimg = cv2.imencode(".jpeg",img, encode_param)

            # GoogleのAPIはbytes、表示するHTMLはbase64の文字列でないといけないらしい
            imageBin = base64.b64encode(bytes(encimg))
            imageString=imageBin.decode()

            payload = {'image': {'image_bytes': bytes(encimg)}}
            client = automl_v1beta1.AutoMlClient.from_service_account_json(Params.keypath)
            prediction_client = automl_v1beta1.PredictionServiceClient.from_service_account_json(Params.keypath)

            params = {"score_threshold": bytes(b'0.5')}
            model_full_id = client.model_path(Params.project_id, Params.compute_region, Params.model_id)
            response = prediction_client.predict(model_full_id, payload,params)

            # 地味にここがミソでクラスの詳細がドキュメントにかかれていないので苦労した
            response=response.payload[0]

            resultString = "ある" if result == 1 else "ない"

            # 画像を含んだ結果をHTMLに埋め込む
            # 画像を埋め込んだ理由はCloud FunctionsからStorageにアップロードできないように作られているためである
            # （不正なアップローダー防止の対策とはいえ、めんどくさい仕様だ・・・
            return render_template("result.html",
                                   image_string=imageString,
                                   class_name=f"class_{result}", result=resultString)

        # GETなどの例外処理
        return render_template("error.html", reason="想定されていないため")

# 参考URL
# https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
# https://cloud.google.com/vision/automl/docs/base64?hl=ja
# https://cloud.google.com/vision/automl/docs/predict?hl=ja#automl-nl-example-python
# https://qiita.com/iss-f/items/fcc766fca27f3685025d
