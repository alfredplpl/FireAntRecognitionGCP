import base64
import cv2
import numpy as np
import traceback

import Params

from google.cloud import automl_v1beta1

# ペイロードの作成。 実はBase64にエンコードしておかないといけないらしい
img = cv2.imread("./joou.jpg")
img = cv2.resize(img, (640, 480))
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
result, encimg = cv2.imencode(".jpg", img, encode_param)
imageBin = base64.b64encode(encimg)
imageString = imageBin.decode()

imageBin=bytes(encimg)
payload = {'image': {'image_bytes': imageBin}}
client = automl_v1beta1.AutoMlClient.from_service_account_json('projectkey.json')
prediction_client = automl_v1beta1.PredictionServiceClient.from_service_account_json('projectkey.json')

params = {"score_threshold": bytes(b'0.5')}
model_full_id = client.model_path(Params.project_id, Params.compute_region, Params.model_id)
response = prediction_client.predict(model_full_id, payload, params)

print( response)

if(response.payload[0].display_name=="fire_ant"):
    print("ヒアリだー！")
else:
    print("ヒアリだー！")
