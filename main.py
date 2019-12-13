# https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
# https://cloud.google.com/vision/automl/docs/base64?hl=ja
# https://cloud.google.com/vision/automl/docs/predict?hl=ja#automl-nl-example-python

def recognize(request):
    #from google.cloud import automl_v1beta1 as automl
    import base64
    import Params

    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'file' not in request.files:
            with open("./htmls/error.html", "r") as f:
                result = f.read()
            return result

        # データの取り出し
        file = request.files['file']

        # ファイル名がなかった時の処理
        if file.filename == '':
            with open("./htmls/error.html", "r") as f:
                result = f.read()
            return result
        # ファイルのチェック
        if file and allwed_file(file.filename):
            automl_client = automl.AutoMlClient()
            model_full_id = automl_client.model_path(
                Params.project_id, Params.compute_region, Params.model_id
            )

            # ペイロードの作成。 実はBase64にエンコードしておかないといけないらしい
            imageString = base64.b64encode(file.data)
            payload = {"image": {"image_bytes": imageString}}

            prediction_client = automl.PredictionServiceClient()
            response = prediction_client.predict(model_full_id, payload, Params.score_threshold)
            print("Prediction results:")
            result=response.payload

            with open("./htmls/resultTrue.html", "r") as f:
                # 画像を含んだ結果をHTMLに埋め込む
                resultHTML = f.read()
                resultHTML.format(image_string=imageString,class_name=result.display_name,score=result.classification.score)
                return resultHTML

    #
    with open("./htmls/error.html", "r") as f:
        result = f.read()
        return result

def allwed_file(filename):
    # .があるかどうかのチェックと、拡張子の確認
    # OKなら１、だめなら0
    ALLOWED_EXTENSIONS = set(['jpg','jpeg'])
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS