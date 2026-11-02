軌道予測MLPの学習再現

1．同梱データからの再学習
  ルートのREADME.mdに従って.venv-trainingを作成し，ルートで実行します．
    .venv-training/bin/python training/prepare_dataset.py
    .venv-training/bin/python training/train_predictor.py
  outputs/training/dataset.npz：訓練・検証の入力と教師
  outputs/training/dataset.json：使用したエピソードIDと窓数
  outputs/training/trajectory_predictor.npz：NumPy推論用の最良重み
  outputs/training/training.json：学習条件・検証誤差の履歴・ライブラリ版

  訓練84軌道・15624窓，検証21軌道・3906窓です．別の21軌道は学習・重み選択に使いません．
  入力：過去10点の相対XY／0.05 m，相対時刻[s]，有効フラグの計40値
  出力：現在の検出位置を基準とした，0.1～0.5秒先の5点の相対XY／0.05 m
  教師：未来のシミュレータ座標．入力に真値や軌道名・設定値を入れません．
  欠測を含む履歴は除外するため，訓練中の有効フラグは全て1です．
  MLP：40→64→64→10，ReLU，7434パラメータ
  学習：MSE，Adam，学習率0.001，batch 256，300 epoch，seed 27，CPU 1スレッド
  検証MSE最小のepochを採用し，推論はPyTorchを使わずNumPyで計算します．
  元の結果：best_epoch=297（0始まり），検証座標RMSE=4.615884827 mm
  この値は将来XY座標の誤差で，把持の成功率ではありません．
  学習はCPUで約9秒でした．環境により所要時間や浮動小数の結果は変わり得ます．
  再現確認：PyTorch 2.7.1+cu126をCPUで使用，NumPy 1.26.4
  CPU専用wheelや他CPUでのビット単位一致までは確認していません．

2．画像からデータを再収集する場合
  MuJoCo用.venvを構築し，ルートで実行します．ROSノード・VLAサーバーの起動は不要です．
    source env.sh
    .venv/bin/python training/collect_dataset.py
    .venv-training/bin/python training/prepare_dataset.py --data outputs/collected --output outputs/recollected_training
    .venv-training/bin/python training/train_predictor.py --output outputs/recollected_training
  7軌道×18条件=126本，1本20秒・10 Hz相当の25200画像を描画します．
  設定生成のseedは31，最大速度は5/10/20 cm/sです．
  腕は初期姿勢に固定し，物体を設定位置に置いてオフライン描画します．
  把持動作や物体遮蔽を含む学習データではありません．動画・画像自体は保存しません．
  赤色重心と既知4点の平面校正から検出XYを得て，真値と別の配列に保存します．
  outputs/collected/に校正・計画・各エピソードができ，存在するエピソードは再描画しません．
  --limit 1は描画確認用です．全126本がないデータは学習用に使わないでください．

3．再学習した重みで把持を実行する場合
  同梱版を残したまま試すには，別のチェックアウトで以下を行います．
    cp outputs/training/trajectory_predictor.npz assets/trajectory_predictor.npz
  カメラ条件を変えて再収集した場合は，新しい校正も組にして使います．
    cp outputs/collected/calibration.npz assets/camera_calibration.npz
  その後，ルートのREADME.mdの3端末の手順で実行します．
