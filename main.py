def recognizeByAutoKeras(request):
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

            # center cropping
            w = img.shape[1]
            h = img.shape[0]
            edge = np.min(img.shape)
            img = img[(h - edge) // 2:(h + edge) // 2, (w - edge) // 2:(w + edge) // 2, :]
            img = cv2.resize(img, (224, 224))

            img = img[np.newaxis, :, :, :]
            clf = pickle_from_file(Params.model_path)
            result = clf.predict(img)
            result=int(result[0])
            resultString="ある" if result==1 else "ない"

            # 画像を含んだ結果をHTMLに埋め込む
            # 画像を埋め込んだ理由はCloud FunctionsからStorageにアップロードできないように作られているためである
            # （不正なアップローダー防止の対策とはいえ、めんどくさい仕様だ・・・
            return render_template( "result.html",
                                    image_string=imageString,
                                    class_name=f"class_{result}",result=resultString)

    # GETなどの例外処理
    return render_template("error.html", reason="想定されていないため")
