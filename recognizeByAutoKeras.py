def recognizeByAutoKeras(request):
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
            from autokeras.utils import pickle_from_file

            # 非公開なパラメータを入れておくところ
            import Params

            MAX_WIDTH=640
            MAX_HEIGHT=480

            # ペイロードの作成。大きすぎる画像をリサイズ。
            # 画像をはbase64にエンコードしておく
            img_array = np.asarray(bytearray(file.stream.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, 1)

            if(img.shape[0]>MAX_HEIGHT):
                img= cv2.resize(img,(int(img.shape[1]*MAX_WIDTH//img.shape[0]),MAX_HEIGHT))
            if(img.shape[1]>MAX_WIDTH):
                img= cv2.resize(img,(MAX_WIDTH,int(img.shape[0]*MAX_HEIGHT//img.shape[1])))

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            result, encimg = cv2.imencode(".jpeg",img, encode_param)
            imageBin = base64.b64encode(bytes(encimg))
            imageString=imageBin.decode()

            # AutoKerasで作ったモデルで画像を判定する
            img = cv2.imdecode(img_array, 1)
            img = img[np.newaxis, :, :, :]
            clf = pickle_from_file(Params.model_path)
            result = clf.predict(img)
            result=int(result[0])

            with open("./htmls/result.html", "r") as f:
                # 画像を含んだ結果をHTMLに埋め込む
                # 画像を埋め込んだ理由はCloud FunctionsからStorageにアップロードできないように作られているためである
                # （不正なアップローダー防止の対策とはいえ、めんどくさい仕様だ・・・
                resultHTML = f.read()
                resultString="ある" if result==1 else "ない"
                resultHTML = resultHTML.format(image_string=imageString, class_name=f"class_{result}",
                                               score="NaN",result=resultString)

                return resultHTML

    # GETなどの例外処理
    with open("./htmls/error.html", "r") as f:
        resultHTML = f.read()
        resultHTML = resultHTML.format(reason="想定されていないため")
        return resultHTML