本課題のMuJoCo描画画像から収集した数値データです．
公開ALOHA実演データセット本体ではありません．

trajectories/plan.json：126エピソードの軌道設定と分割
trajectories/episodeNNN.npz：各20秒・10 Hz，200点
  times：時刻[s]
  observed：画像の赤色重心を平面校正したXY[m]．予測器の入力に使用
  truth：シミュレーションで設定したXY[m]．教師のみに使用
  detection_ms：画像位置検出の処理時間[ms]．学習入力には使用しない
trajectories/calibration.npz：既知4点と画像重心による平面校正

7種類それぞれ訓練12本・検証3本・テスト3本で，計84/21/21本です．
同じエピソードの窓が訓練と検証の両方へ入らないよう分けています．
再収集方法はtraining/README.txtを参照してください．
