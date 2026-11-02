camera_calibration.npz
  data/trajectories/calibration.npzと同じ校正です．
  固定topカメラ・既知高さで，赤色重心を机上XYへ変換します．
  再生成：training/collect_dataset.py
  SHA256: fbcbd34c8a36259617e1fffb5a28f73b00ef842d865c4aa5ad0ef13489541fe1

trajectory_predictor.npz
  本課題で教師あり学習した7434パラメータの軌道予測MLPです．
  再学習：training/prepare_dataset.py → training/train_predictor.py
  重みw0/b0，w1/b1，w2/b2をsrc/object_tracker.pyが読み込みます．
  元の学習ログはresults/training.json，再現確認はresults/reproduction.jsonです．
  SHA256: 0c4eef5dd15f2bc9612330729f0666da7f98c6ff9136ab6880579b4e3db07ac2

同梱の数値ファイルは本課題で生成しました．VLA重みや外部ライブラリ本体は含みません．
