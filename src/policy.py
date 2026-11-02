"""公式OpenPIサーバーに画像と関節状態を送り，関節目標値を受け取る"""

from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy


def connect(host="localhost", port=8000):
    return WebsocketClientPolicy(host=host, port=port)


def predict(client, state, image, horizon=10):
    # 公式ALOHA例と同じく，縦横比を保って224角へ余白を追加し，CHW順にする
    image = image_tools.resize_with_pad(image, 224, 224).transpose(2, 0, 1)
    result = client.infer({"state": state, "images": {"cam_high": image}})
    # モデルは50点を予測し，その先頭から指定数だけ実行する
    return result["actions"][:horizon]
