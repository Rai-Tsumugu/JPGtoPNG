# JPG → SVG → PNG  色忠実変換パイプライン

JPEG の圧縮アーティファクト（ジャギー・中間色）を排除し、
使用色を確定してから SVG・PNG を生成します。
JPG（量子化済み）・SVG・PNG の 3 出力は同一カラーパレットを共有します。

## パイプライン

```
JPEG
 │
 ├─[1] 読み込み & ノイズ除去
 │       双方向フィルタ + メジアンフィルタで DCT アーティファクトを除去
 │
 ├─[2] 色量子化（CIE Lab K-means）
 │       ジャギーによる中間色を排除し、使用色を確定
 │       近似色クラスタは CIE76 距離で自動統合
 │       -> output/{name}.jpg  ← 同パレット
 │
 ├─[3] ベクター化 → SVG
 │       量子化画像をトレース（直線・曲線を検出）
 │       バックエンド: vtracer（推奨）→ pipeline 自前実装
 │       -> output/{name}.svg  ← 同パレット
 │
 └─[4] ラスタライズ → PNG
         SVG を元解像度で PNG に変換
         バックエンド: cairosvg → svglib+reportlab → Inkscape → ImageMagick
         -> output/{name}.png  ← 同パレット
```

## クイックスタート

```bat
run.bat logo.jpg
```

または:

```bash
python convert.py logo.jpg
```

`input/` に置いたファイルはファイル名だけで指定できます:

```bat
run.bat logo_HUmeta.jpg
```

## インストール

```bash
pip install -r requirements.txt
```

**ベクター化（Stage 3）— 推奨:**
```bash
pip install vtracer      # Rust ベース・高速・高品質
```

**ラスタライズ（Stage 4）— 推奨:**
```bash
pip install cairosvg     # 高品質（Cairo ライブラリが必要）
# または
pip install svglib reportlab  # 純 Python フォールバック
```

## 使い方

```bat
rem 全出力: output/{name}.jpg + .svg + .png
run.bat logo.jpg

rem 色数を絞る（シンプルなロゴ向け）
run.bat logo.jpg -k 8

rem SVG のみ出力
run.bat logo.jpg --svg-only

rem PNG のみ出力（SVG は一時ファイル）
run.bat logo.jpg --png-only

rem 量子化 JPG のみ出力（色確認用）
run.bat logo.jpg --jpg-only

rem 出力ディレクトリを指定
run.bat logo.jpg -o my_output/

rem サイレント実行
run.bat logo.jpg -q
```

## オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `-o, --output` | `output/` | 出力ディレクトリ |
| `--jpg-only` | off | 量子化 JPG のみ出力 |
| `--svg-only` | off | SVG のみ出力 |
| `--png-only` | off | PNG のみ出力（SVG は一時ファイル） |
| **色量子化** | | |
| `-k, --clusters N` | `24` | 初期色クラスタ数（少ないほどシンプルな色） |
| `--merge-threshold D` | `12.0` | 類似色統合の CIE76 距離閾値 |
| `--denoise D` | `5` | JPEG ノイズ除去強度（0=無効、最大 15） |
| **ベクター化** *(vtracer 非使用時のみ有効)* | | |
| `--epsilon E` | `0.002` | RDP 簡略化比率（大きいほど少ない制御点） |
| `--no-curves` | off | Bezier 曲線を使わずポリラインで出力 |
| `--morph-kernel K` | `3` | 形態学カーネルサイズ |
| `-q, --quiet` | off | 進捗出力を抑制 |

## ファイル構成

```
JPGtoPNG/
├── convert.py              # メインパイプライン（唯一のエントリポイント）
├── run.bat                 # Windows 用ランチャー
├── requirements.txt        # 依存パッケージ
├── input/                  # 入力 JPEG を置く場所
├── output/                 # 出力ファイル（jpg / svg / png）
├── stages/
│   ├── clustering.py       # [2] CIE Lab K-means 色クラスタリング
│   ├── masking.py          # [3] クラスタ別バイナリマスク生成
│   ├── contour_extractor.py# [3] 輪郭抽出 + RDP 簡略化
│   ├── vectorizer.py       # [3] Catmull-Rom Bezier → SVG
│   └── renderer.py         # （予約）
└── utils/
    ├── bezier.py            # RDP / Catmull-Rom / SVG パス生成
    └── image_utils.py       # 画像 I/O・前処理
```

## ベクター化バックエンドについて

### vtracer（推奨）
`pip install vtracer` でインストール可能な Rust 製ライブラリ。
カラーモードで量子化画像を直接トレースし、色・形状ともに高品質な SVG を生成します。

### pipeline 自前実装（フォールバック）
vtracer が利用できない場合に使用:
- クラスタ別バイナリマスクから `cv2.findContours` で輪郭を抽出
- Catmull-Rom スプライン → Cubic Bezier 変換で滑らかな曲線を生成
- `svgwrite` で SVG として出力

## 動作環境

- Python 3.10+
- Windows / macOS / Linux
- 必須: `opencv-python`, `numpy`, `scikit-learn`, `Pillow`, `svgwrite`, `scipy`, `scikit-image`
