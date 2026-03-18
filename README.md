# JPEG → Complete PNG Restoration Pipeline

研究レベルの JPEG → PNG 再構成パイプライン。
JPEG 圧縮アーティファクトを排除し、ベクター中間表現を経由してクリーンな PNG を生成します。

## パイプライン概要

```
JPEG
 │
 ├─[1] 前処理          双方向フィルタ + メジアンフィルタ (DCT ノイズ除去)
 ├─[2] 色クラスタリング  MiniBatch K-means in CIE Lab 空間
 ├─[3] マスク生成       クラスタ別バイナリマスク + 形態学的クリーンアップ
 ├─[4] 輪郭抽出         cv2.findContours + RDP 簡略化
 ├─[5] ベクター化        Catmull-Rom Bezier 曲線 → SVG
 └─[6] 再レンダリング    超解像 OpenCV fillPoly → 最終 PNG
```

## ファイル構成

```
JPGtoPNG/
├── pipeline.py              # CLI エントリポイント・パイプライン制御
├── config.py                # PipelineConfig 全パラメータ定義
├── requirements.txt         # 依存パッケージ
├── stages/
│   ├── clustering.py        # Stage 2: 色クラスタリング
│   ├── masking.py           # Stage 3: マスク生成
│   ├── contour_extractor.py # Stage 4: 輪郭抽出
│   ├── vectorizer.py        # Stage 5: SVG ベクター化
│   └── renderer.py          # Stage 6: PNG 再レンダリング
└── utils/
    ├── bezier.py             # RDP / Catmull-Rom / SVG パス生成
    └── image_utils.py        # 画像 I/O・前処理・品質指標
```

## インストール

```bash
pip install -r requirements.txt
```

> **オプション:** より高品質な SVG→PNG レンダリングが必要な場合は [Cairo](https://cairographics.org/) をインストールした後 `pip install cairosvg` を追加してください。

## 使い方

### 基本

```bash
python pipeline.py input.jpg
# -> input_restored.png  および各中間ファイルを同じディレクトリに出力
```

### 出力先を指定

```bash
python pipeline.py input.jpg -o output/clean.png
```

### パラメータ調整

```bash
# 色数を増やして精細に復元
python pipeline.py input.jpg --clusters 48

# 輪郭をよりシャープに (RDP を弱める)
python pipeline.py input.jpg --epsilon 0.0005

# 超解像倍率を上げてアンチエイリアス強化
python pipeline.py input.jpg --scale 4

# ベクター曲線なし (ポリライン, デバッグ用)
python pipeline.py input.jpg --no-curves

# 中間ファイル保存なし・サイレント実行
python pipeline.py input.jpg --no-intermediates -q
```

## オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `-o, --output` | `<stem>_restored.png` | 出力 PNG パス |
| `-k, --clusters` | `24` | 初期色クラスタ数 |
| `--merge-threshold` | `12.0` | CIE76 距離でのクラスタ統合閾値 |
| `--morph-kernel` | `3` | 形態学カーネルサイズ |
| `--epsilon` | `0.002` | RDP 簡略化比率 (弧長に対する割合) |
| `--no-curves` | off | Bezier 曲線を使わずポリラインで出力 |
| `--tension` | `1.0` | Catmull-Rom テンション (低いほど緩やか) |
| `--scale` | `2` | 超解像スケール (アンチエイリアス用) |
| `--cairosvg` | off | cairosvg で SVG→PNG レンダリング |
| `--width / --height` | `0` (入力と同じ) | 出力サイズ上書き |
| `--denoise` | `5` | ノイズ除去強度 (0 = 無効) |
| `--no-intermediates` | off | 中間ファイルを保存しない |
| `-q, --quiet` | off | 進捗出力を抑制 |

## 中間出力ファイル

`--no-intermediates` を指定しない場合、以下のファイルが出力されます:

| ファイル名 | 内容 |
|---|---|
| `<stem>_1_preprocessed.png` | JPEG ノイズ除去後の画像 |
| `<stem>_2_quantized.png` | 色クラスタリング後の量子化画像 |
| `<stem>_3_masks_composite.png` | 全クラスタマスクの合成デバッグ画像 |
| `<stem>_vector.svg` | Bezier 曲線ベクターファイル |
| `<stem>_restored.png` | 最終出力 PNG |

## アルゴリズムの詳細

### 色クラスタリング (Stage 2)
- **CIE Lab 色空間**を使用。RGB と異なり、等距離が等知覚差に対応するため JPEG 色ズレに強い。
- MiniBatch K-means で大画像でも高速処理。
- 隣接クラスタの CIE76 距離が閾値以下なら Union-Find で自動統合。

### 輪郭簡略化 (Stage 4)
- RDP (Ramer-Douglas-Peucker) の ε を**輪郭弧長の比率**で指定することで画像サイズに依存しない自動スケーリング。

### ベクター化 (Stage 5)
- **Catmull-Rom スプライン → Cubic Bezier** 変換により全制御点を通過する滑らかな曲線を生成。
- 変換式: `CP1 = P[i] + (P[i+1] - P[i-1]) / 6 × tension`
- SVG の `C` コマンドとして出力。

### 再レンダリング (Stage 6)
- `render_scale` 倍の解像度でポリゴン塗り潰し後、`INTER_AREA` で縮小してアンチエイリアスを実現。
- cairosvg が利用可能なら SVG 直接ラスタライズに切り替え。

## 品質指標

パイプライン終了時に元の JPEG との比較指標を表示します:

- **SSIM** (Structural Similarity): 1.0 = 完全一致
- **PSNR**: 高いほど元画像に近い (単位: dB)

> 目的は「JPEG の完全復元」ではなく「アーティファクトのないクリーン画像の生成」のため、SSIM/PSNR は参考値として扱ってください。

## 動作環境

- Python 3.10+
- Windows / macOS / Linux
- 依存: opencv-python, numpy, scikit-learn, Pillow, svgwrite, scipy, scikit-image
