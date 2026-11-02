# 軌道予測MLPによるVLAの動的把持補正

画像履歴からブロックの未来位置を予測し，VLAの動作指令を補正することで，移動するブロックの把持を試みるプロジェクトです．Physical AI講座の最終課題として，ALOHAのMuJoCoシミュレーションとπ0，ROS 2を用いて実装しました．

VLAの推論と指令列の実行には時間がかかり，その間にも対象物は移動します．本実装では，VLAの後段に軌道予測と手先位置の補正処理を追加し，画像観測10 Hzに対して関節指令を50 Hzで更新します．

## 手法

1. topカメラの画像と実関節状態をπ0に入力し，関節目標の列を生成する
2. 画像中の赤いブロックを検出し，過去10点の位置履歴からMLPで0.1～0.5秒先のXY位置を予測する
3. 予測位置を使ってVLAの手先目標を補正し，微分IKで右腕の関節目標へ変換する
4. 両腕の関節指令を連続化し，ROS 2経由でシミュレーションを制御する

| 処理 | 周期・設定 |
| --- | --- |
| 画像取得・物体位置検出・MLP予測 | 10 Hz |
| π0推論 | 実測約2.5 Hz．50点予測の先頭10点を実行 |
| 後段補正・IK・関節指令 | 50 Hz |
| MuJoCo内部積分 | 2 ms |

観測間は予測した軌道列を時間補間します．物体の真値や軌道設定は補正器へ渡さず，画像から得た位置と実関節状態を使います．MLPは未来の物体位置を教師あり学習したもので，π0の追加学習や強化学習は行っていません．詳細は[構成と入出力](docs/architecture.txt)を参照してください．

## 実験結果

固定した動的11条件で，補正の有無を比較しました．

| 方式 | 持ち上げ到達 |
| --- | ---: |
| VLA・補正なし | 0/11 |
| VLA＋軌道予測MLPによる補正 | 8/11（72.7%） |

**8件中2件は，把持後短時間で落下した不安定な把持です．** 残る6件は1秒保持も満たしました．

持ち上げ到達は，軌道拘束解除後，最初の落下・拾い直しの前に「ブロック中心の高さ7 cm以上」と「右腕両指の接触」を記録したことと定義します．把持後の受け渡し完了は条件に含めません．両方式に同じ関節指令の連続化を適用しています．各試行の条件・判定は[評価結果](results/grasping.json)，検証範囲は[結果ファイルの説明](results/README.txt)に記載しています．

固定カメラ・既知平面・単一の赤いブロックを前提とした検証です．学習データは腕を初期姿勢に固定して収集しており，把持中の遮蔽や失敗後の再追跡は今後の課題です．

## ディレクトリ構成

```text
src/          ROS 2ノード，シミュレーション，画像認識，予測・補正
training/     学習データ収集・前処理・MLP学習
data/         画像検出位置・教師位置・軌道設定
assets/       学習済みMLPとカメラ校正
configs/      ブロック軌道の設定例
results/      学習ログ・把持評価・再現確認結果
docs/         各ファイルの役割とモデルの入出力
```

以降のコマンドはリポジトリのルートで実行します．

## MLPの学習再現

同梱の数値データからCPUで学習できます．ROS・MuJoCo・GPUは不要です．Python環境の作成にはuvを使います．

```bash
source env.sh
uv venv --python 3.11 .venv-training
uv pip install --python .venv-training/bin/python -r requirements-training.txt --index-url https://download.pytorch.org/whl/cpu
.venv-training/bin/python training/prepare_dataset.py
.venv-training/bin/python training/train_predictor.py
```

学習結果は`outputs/training/`へ保存します．同梱の実行用重みは上書きしません．訓練84軌道，検証21軌道を使用し，MLPは40→64→64→10の7434パラメータです．

データの再収集，学習条件，重みの切り替えは[学習手順](training/README.txt)を参照してください．

## シミュレーションの環境構築

動作確認環境はUbuntu 24.04，ROS 2 Jazzy，RTX 4070 Ti SUPERです．以下が導入済みであることを前提とします．

- ROS 2の`rclpy`・`cv_bridge`・`message_filters`・標準メッセージ
- `python3-opencv`（確認時4.6.0）
- uv，git，GPUドライバ

ROSノードはPython 3.12，OpenPIサーバーはPython 3.11を使用します．OpenCVは`system-site-packages`から参照し，pip版で上書きしません．

```bash
source /opt/ros/jazzy/setup.bash
source env.sh
mkdir -p third_party
git clone https://github.com/Physical-Intelligence/openpi.git third_party/openpi
git -C third_party/openpi checkout 215abfb217dbac7d5f1273282331b9b1866c0479
GIT_LFS_SKIP_SMUDGE=1 uv sync --project third_party/openpi --frozen --no-dev --python 3.11
uv venv --system-site-packages --python /usr/bin/python3 .venv
uv pip install --python .venv/bin/python -r requirements.txt -e third_party/openpi/packages/openpi-client
```

## 実行

3つの端末を開き，各端末で次を実行します．

```bash
source /opt/ros/jazzy/setup.bash
source env.sh
```

### 1．π0推論サーバー

```bash
third_party/openpi/.venv/bin/python third_party/openpi/scripts/serve_policy.py --port 8000 --env ALOHA_SIM
```

初回は公式チェックポイントを取得します（約12 GB）．保存済みの重みを使う場合は，`--env ALOHA_SIM`を次に置き換えます．

```text
policy:checkpoint --policy.config pi0_aloha_sim --policy.dir checkpoints/pi0_aloha_sim
```

### 2．VLAのROSノード

```bash
.venv/bin/python src/policy_node.py --chunk-size 10
```

初回コンパイルが終わり，「π0の準備が完了しました」と表示されるまで待ちます．このノードは停止するまで次の試行も受け付けます．

### 3．把持シミュレーション

軌道予測MLPで補正しながら，リサージュ軌道を動くブロックを把持します．

```bash
.venv/bin/python src/sim_node.py --trajectory configs/lissajous12.json --correction mlp --seconds 20 --video outputs/dynamic_mlp.mp4
```

動画と同名のNPZに関節指令・実関節状態・画像検出位置・評価用の物体真値などを保存します．推論中もシミュレーションは進行します．実時間に対する処理の遅れは，終了時の`sim`と`wall`で確認できます．

### 補正なし・静止条件との比較

```bash
.venv/bin/python src/sim_node.py --trajectory configs/lissajous12.json --correction none --seconds 20 --video outputs/dynamic_none.mp4
.venv/bin/python src/sim_node.py --seed 1 --correction none --seconds 20 --video outputs/static.mp4
```

`--correction none`は物体追従の補正を外し，両腕の指令連続化を残します．`--correction`自体を省略すると連続化もない直接実行になるため，比較時は`none`を明示します．位置保持と等速予測は，それぞれ`position`と`velocity`で指定できます．

上記は単一条件の実行例です．報告した11条件の設定は[評価結果](results/grasping.json)を参照してください．

## ブロック軌道の設定

`--trajectory`にJSONファイルを指定します．設定例は[configs/](configs/)にあります．

| 項目 | 意味 |
| --- | --- |
| `kind` | `x_line`，`y_line`，`diagonal`，`circle`，`square`，`lissajous12`，`lissajous34` |
| `center` | XY中心[m] |
| `radius` | 直線の振幅・円の半径・四角の半辺[m] |
| `speed` | 最大XY速度[m/s]．0なら中心で静止 |
| `phase` | 開始位相[rad] |
| `direction` | 進行方向（1または−1） |
| `delta` | リサージュの位相差[rad] |
| `angle` | 斜め直線の方向[rad] |

同じ腕の両指接触が3制御ステップ続くと軌道拘束を解除し，以降は物理シミュレーションに任せます．接触真値や解除フラグは補正器へ渡しません．

## 使用リソース

- シミュレーション：[gym-aloha](https://github.com/huggingface/gym-aloha)
- VLA・公式実行例：[OpenPI / ALOHA Sim](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479/examples/aloha_sim)
- 元タスクの公開データ：[ALOHA Sim Transfer Cube Human](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human)
- 公式VLA重み：`gs://openpi-assets/checkpoints/pi0_aloha_sim`

各コード・データ・重みには配布元の利用条件が適用されます．OpenPIコードとVLA重みの条件は同一ではなく，Gemma由来の条件は[Gemma Terms of Use](https://ai.google.dev/gemma/terms)も参照してください．

本リポジトリの`data/`と`assets/`には，本課題で生成した軌道データ・小型MLP重み・カメラ校正を収録しています．
