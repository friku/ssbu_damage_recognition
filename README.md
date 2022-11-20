# ssbu_damage_recognition

## 学習方法
config.ymlのtrain_dataset_dirにデータセットへのpathを指定する。
指定したデータセットのディレクトリは00~09,10_hit,11_nullの計12ディレクトリがあり、各ディレクトリには数字の画像と数字が変わる瞬間の画像、数字が写っていない画像が入っている。この画像はcut_damage_area.pyで切り取った画像を手動で振り分けて作成する。
config.ymlのtrain_nameを変更する。train_nameが重みの名前になる。
```
python train_damage.py
```

## 推論方法
config.ymlのtest_weight_nameを推論時に使用する重みの名前を指定する。
config.ymlのtest_movie_pathに推論させたい動画のパスを指定する。
```
python detect_damage.py
```

