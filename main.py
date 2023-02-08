import base64
import cv2 as cv
import numpy as np
from flask import jsonify, Flask, request, render_template, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)


def base64_to_img(base64_str):
    # 传入为RGB格式下的base64，传出为RGB格式的numpy矩阵
    byte_data = base64.b64decode(base64_str)  # 将base64转换为二进制
    encode_image = np.asarray(bytearray(byte_data), dtype="uint8")  # 二进制转换为一维数组
    img_array = cv.imdecode(encode_image, cv.IMREAD_COLOR)  # 用cv2解码为三通道矩阵
    img_array = cv.cvtColor(img_array, cv.COLOR_BGR2RGB)  # BGR2RGB
    return img_array


def img_to_base64(img_array):
    # 传入图片为RGB格式numpy矩阵，传出的base64也是通过RGB的编码
    img_array = cv.cvtColor(img_array, cv.COLOR_RGB2BGR)  # RGB2BGR，用于cv2编码
    encode_image = cv.imencode(".jpg", img_array)[1]  # 用cv2压缩/编码，转为一维数组
    byte_data = encode_image.tobytes()  # 转换为二进制
    base64_str = base64.b64encode(byte_data).decode("ascii")  # 转换为base64
    return base64_str


@app.route('/')
def index_html():
    return render_template('realtime_camera.html')


@app.route('/get_img', methods=['POST'])
def receive_pic():
    # receive the base64 image
    img_base64 = request.form.get('img')[len("data:image/png;base64,"):]

    # transform the base64 image to normal image
    result = base64_to_img(img_base64)
    print(result[0, 0])

    # TODO: use the model to detect the image, input: `result`<numpy img(RGB)>, output: `result`<numpy img(RGB)>
    # model_path = ""
    # config_path = ""
    # model = torch.load(model_path, config_path)
    # result = model(result)

    # return the detected image
    base64_img = img_to_base64(result)

    respose = {
        "code": 200,
        "base64_img": "data:image/png;base64," + str(base64_img)
    }
    # compare
    # print("origin:", img_base64[0:50])
    # print("return:", base64_img[0:50])

    return jsonify(respose)


if __name__ == "__main__":
    # app.run(host="127.0.0.1", port=5006, debug=False)
    app.run(host="0.0.0.0", port=5006, debug=False, ssl_context='adhoc')
