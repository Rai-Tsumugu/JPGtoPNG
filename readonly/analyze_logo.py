#!/usr/bin/env python3
"""
logo_HUmeta.jpg 解析スクリプト（読み取り専用）
================================================

8000px超の大型JPEG画像をタイル分割して解析する。
原画像は一切変更しない。

解析内容:
  1. 色分析  — JPEGジャギーの中間色を除去し、4色パレットに量子化
                 (背景:灰色, ロゴ:青・赤・緑)
  2. 形状分析 — 直線部分と曲線部分を検出・分類

出力先: readonly/output/
  quantized.png   — パレット量子化画像
  geometry.png    — 形状マップ (青=直線, オレンジ=曲線)
  report.json     — 統計レポート

使用方法:
  python readonly/analyze_logo.py input/logo_HUmeta.jpg
  python readonly/analyze_logo.py input/logo_HUmeta.jpg --scale 0.5
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# ─────────────────────────────────────────────────────────────────────────────
# 定数
# ─────────────────────────────────────────────────────────────────────────────

TILE_W = 2000        # タイル幅  (px)
TILE_H = 2000        # タイル高さ (px)
SAMPLE_STEP = 8      # 色サンプリング間隔 (ピクセル数)
N_COLORS = 4         # パレット色数 (背景灰+青+赤+緑)

# HoughLinesP パラメータ (2000px タイル基準)
HOUGH_THRESHOLD   = 40    # 累積カウント閾値
HOUGH_MIN_LEN     = 25    # 最短線分長 (px)
HOUGH_MAX_GAP     = 6     # 線分間の最大隙間 (px)


# ─────────────────────────────────────────────────────────────────────────────
# タイル読み込み
# ─────────────────────────────────────────────────────────────────────────────

def iter_tiles(path: Path):
    """
    大型画像をタイルに分割して yield する。
    PIL は open 時にピクセル全体をロードしないため、
    8000px 超でもメモリ効率よく処理できる。

    Yields:
        (tile_rgb: np.ndarray, x: int, y: int, x2: int, y2: int)
    """
    img = Image.open(path)
    W, H = img.size
    for y in range(0, H, TILE_H):
        for x in range(0, W, TILE_W):
            x2 = min(x + TILE_W, W)
            y2 = min(y + TILE_H, H)
            tile = np.array(img.crop((x, y, x2, y2)))
            if tile.ndim == 2:                        # グレースケール → RGB
                tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)
            elif tile.shape[2] == 4:                  # RGBA → RGB
                tile = cv2.cvtColor(tile, cv2.COLOR_RGBA2RGB)
            yield tile, x, y, x2, y2
    img.close()


# ─────────────────────────────────────────────────────────────────────────────
# 色分析
# ─────────────────────────────────────────────────────────────────────────────

def sample_colors(path: Path) -> np.ndarray:
    """全タイルから間引きサンプリングして LAB 色空間の配列を返す。"""
    samples = []
    img = Image.open(path)
    W, H = img.size
    for y in range(0, H, TILE_H):
        for x in range(0, W, TILE_W):
            x2 = min(x + TILE_W, W)
            y2 = min(y + TILE_H, H)
            tile = np.array(img.crop((x, y, x2, y2)))
            if tile.ndim == 2:
                tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)
            elif tile.shape[2] == 4:
                tile = tile[:, :, :3]
            tile_lab = cv2.cvtColor(tile, cv2.COLOR_RGB2LAB)
            flat = tile_lab[::SAMPLE_STEP, ::SAMPLE_STEP].reshape(-1, 3)
            samples.append(flat)
    img.close()
    return np.vstack(samples).astype(np.float32)


def find_palette(samples: np.ndarray, n_colors: int = N_COLORS):
    """
    KMeans (LAB空間) でパレット色を検出する。

    Returns:
        centers_lab : np.ndarray  shape (n, 3)  LAB値
        centers_rgb : np.ndarray  shape (n, 3)  RGB値 uint8
    """
    km = KMeans(n_clusters=n_colors, n_init=15, random_state=42)
    km.fit(samples)
    centers_lab = km.cluster_centers_                      # float64

    # LAB → RGB 変換
    lab_u8 = np.clip(centers_lab, 0, 255).astype(np.uint8).reshape(1, n_colors, 3)
    rgb_u8 = cv2.cvtColor(lab_u8, cv2.COLOR_LAB2RGB)[0]   # (n, 3) uint8
    return centers_lab, rgb_u8


def label_color(rgb: np.ndarray) -> str:
    """RGB から色ラベルを推定する。"""
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    sat = max(r, g, b) - min(r, g, b)

    if sat < 35:
        return "gray"
    if r >= g and r >= b:
        return "red"
    if b >= r and b >= g:
        return "blue"
    if g >= r and g >= b:
        return "green"
    return f"unknown(R{r}G{g}B{b})"


# ─────────────────────────────────────────────────────────────────────────────
# 量子化
# ─────────────────────────────────────────────────────────────────────────────

def quantize_tile(tile_rgb: np.ndarray, centers_lab: np.ndarray) -> np.ndarray:
    """
    タイルの各ピクセルを最近傍パレット色インデックスにマップする。
    LAB 空間でユークリッド距離を使用（知覚的精度が高い）。

    Returns:
        indices: np.ndarray  shape (H, W)  dtype uint8  値 0..N_COLORS-1
    """
    lab = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    H, W = lab.shape[:2]
    flat = lab.reshape(-1, 3)

    # 各セントロイドとの距離を一括計算
    dists = np.stack(
        [np.sum((flat - c) ** 2, axis=1) for c in centers_lab],
        axis=1
    )  # (N_px, N_COLORS)
    return np.argmin(dists, axis=1).reshape(H, W).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 形状解析
# ─────────────────────────────────────────────────────────────────────────────

def _quantized_to_gray(q_tile: np.ndarray, bg_idx: int) -> np.ndarray:
    """
    量子化インデックス画像 → Canny 用グレースケール変換。
    背景=0、前景色は 85/170/255 の段階値を付与して境界を鮮明にする。
    """
    n = q_tile.max() + 1
    gray_vals = np.zeros(n, dtype=np.uint8)
    rank = 1
    for i in range(n):
        if i != bg_idx:
            gray_vals[i] = min(255, 85 * rank)
            rank += 1
    return gray_vals[q_tile]


def analyze_geometry(q_tile: np.ndarray, bg_idx: int):
    """
    量子化タイルから直線・曲線エッジを分類する。

    処理:
      1. 量子化画像 → グレースケール変換
      2. Canny エッジ検出（色境界を正確に捉える）
      3. HoughLinesP で直線線分を検出 → straight_mask
      4. エッジ - 直線 = 曲線 → curved_mask

    Returns:
        edges        : np.ndarray  全エッジ (0 or 255)
        straight     : np.ndarray  直線エッジ (0 or 255)
        curved       : np.ndarray  曲線エッジ (0 or 255)
    """
    gray = _quantized_to_gray(q_tile, bg_idx)

    # ガウシアンブラーで量子化アーティファクトを軽減
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # 直線検出 (HoughLinesP)
    straight_mask = np.zeros_like(edges)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LEN,
        maxLineGap=HOUGH_MAX_GAP,
    )
    if lines is not None:
        for seg in lines:
            x1, y1, x2, y2 = seg[0]
            cv2.line(straight_mask, (x1, y1), (x2, y2), 255, 2)

    # 直線マスクを少し膨張させてエッジピクセルを包む
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    straight_dilated = cv2.dilate(straight_mask, kernel, iterations=1)

    straight_on_edge = cv2.bitwise_and(edges, straight_dilated)
    curved_on_edge   = cv2.bitwise_and(edges, cv2.bitwise_not(straight_dilated))

    return edges, straight_on_edge, curved_on_edge


# ─────────────────────────────────────────────────────────────────────────────
# 結果保存
# ─────────────────────────────────────────────────────────────────────────────

def save_quantized(quantized_full: np.ndarray, centers_rgb: np.ndarray,
                   out_path: Path, max_side: int = 4000) -> None:
    """
    量子化画像をパレットモード PNG で保存する。
    4色インデックスのためファイルサイズが非常に小さい。
    """
    H, W = quantized_full.shape
    scale = min(1.0, max_side / max(W, H))
    nw, nh = int(W * scale), int(H * scale)

    # パレット定義 (4色 × 3ch = 12バイト, 残りは0埋め)
    flat_palette = []
    for rgb in centers_rgb:
        flat_palette += [int(rgb[0]), int(rgb[1]), int(rgb[2])]
    flat_palette += [0] * (768 - len(flat_palette))

    img_p = Image.fromarray(quantized_full, mode="L").convert("P")
    img_p.putpalette(flat_palette)

    if scale < 1.0:
        img_p = img_p.resize((nw, nh), Image.NEAREST)

    img_p.save(out_path)
    kb = out_path.stat().st_size / 1024
    print(f"  量子化画像 : {out_path}  ({kb:.0f} KB)")


def save_geometry(straight_full: np.ndarray, curved_full: np.ndarray,
                  W: int, H: int, out_path: Path, max_side: int = 4000) -> None:
    """
    形状マップを PNG で保存する。
    青 = 直線エッジ, オレンジ = 曲線エッジ, 黒 = 非エッジ
    """
    geom = np.zeros((H, W, 3), dtype=np.uint8)
    geom[straight_full > 0] = [60, 120, 255]    # 青: 直線
    geom[curved_full > 0]   = [255, 140, 40]    # オレンジ: 曲線

    scale = min(1.0, max_side / max(W, H))
    img = Image.fromarray(geom)
    if scale < 1.0:
        nw, nh = int(W * scale), int(H * scale)
        img = img.resize((nw, nh), Image.LANCZOS)
    img.save(out_path)
    kb = out_path.stat().st_size / 1024
    print(f"  形状マップ  : {out_path}  ({kb:.0f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# メイン処理
# ─────────────────────────────────────────────────────────────────────────────

def process(input_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 画像情報 ──────────────────────────────────────────────────────────────
    _img_info = Image.open(input_path)
    W, H = _img_info.size
    _img_info.close()

    sep = "=" * 58
    print(f"\n{sep}")
    print(f"  logo解析スクリプト  (読み取り専用)")
    print(f"  入力 : {input_path.resolve()}")
    print(f"  出力 : {output_dir.resolve()}/")
    print(f"  サイズ: {W} x {H} px")
    print(f"{sep}")

    total_tiles = math.ceil(H / TILE_H) * math.ceil(W / TILE_W)

    # ── Step 1: 色サンプリング ────────────────────────────────────────────────
    print(f"\n[1/4] 色サンプリング中  (step={SAMPLE_STEP}px)...")
    samples = sample_colors(input_path)
    print(f"  サンプル数: {len(samples):,}")

    # ── Step 2: パレット検出 ───────────────────────────────────────────────────
    print(f"\n[2/4] パレット検出  (KMeans k={N_COLORS}, LAB空間)...")
    centers_lab, centers_rgb = find_palette(samples, N_COLORS)

    labels = [label_color(c) for c in centers_rgb]
    palette_info = []
    for i, (lbl, rgb) in enumerate(zip(labels, centers_rgb)):
        print(f"  色{i}: {lbl:12s}  RGB=({rgb[0]:3d}, {rgb[1]:3d}, {rgb[2]:3d})")
        palette_info.append({"index": i, "label": lbl,
                              "rgb": [int(rgb[0]), int(rgb[1]), int(rgb[2])]})

    bg_idx = next((i for i, l in enumerate(labels) if l == "gray"), 0)
    print(f"  背景インデックス: {bg_idx} ({labels[bg_idx]})")

    # ── Step 3: タイル処理 (量子化 + 形状解析) ────────────────────────────────
    print(f"\n[3/4] タイル処理中  ({total_tiles} タイル)...")

    quantized_full = np.zeros((H, W), dtype=np.uint8)
    straight_full  = np.zeros((H, W), dtype=np.uint8)
    curved_full    = np.zeros((H, W), dtype=np.uint8)

    tile_n = 0
    for tile_rgb, x, y, x2, y2 in iter_tiles(input_path):
        tile_n += 1

        # 量子化
        q = quantize_tile(tile_rgb, centers_lab)
        quantized_full[y:y2, x:x2] = q

        # 形状解析
        _, straight, curved = analyze_geometry(q, bg_idx)
        straight_full[y:y2, x:x2] = straight
        curved_full[y:y2, x:x2]   = curved

        s_px = int(np.count_nonzero(straight))
        c_px = int(np.count_nonzero(curved))
        print(f"  [{tile_n:3d}/{total_tiles}] ({x:4d},{y:4d})-({x2:4d},{y2:4d})"
              f"  直線={s_px:6d}px  曲線={c_px:6d}px")

    # ── Step 4: 保存 ──────────────────────────────────────────────────────────
    print("\n[4/4] 結果を保存中...")

    save_quantized(quantized_full, centers_rgb,
                   output_dir / "quantized.png")
    save_geometry(straight_full, curved_full, W, H,
                  output_dir / "geometry.png")

    # 統計
    total_px   = W * H
    edge_total = int(np.count_nonzero(straight_full | curved_full))
    s_total    = int(np.count_nonzero(straight_full))
    c_total    = int(np.count_nonzero(curved_full))

    color_dist = {}
    for info in palette_info:
        cnt = int(np.sum(quantized_full == info["index"]))
        color_dist[info["label"]] = {
            "pixel_count": cnt,
            "percent": round(100.0 * cnt / total_px, 2),
            "rgb": info["rgb"],
        }

    report = {
        "image": str(input_path),
        "size": {"width": W, "height": H},
        "tile_size": {"width": TILE_W, "height": TILE_H},
        "palette": palette_info,
        "color_distribution": color_dist,
        "geometry": {
            "total_edge_px": edge_total,
            "straight_px": s_total,
            "curved_px": c_total,
            "straight_ratio": round(s_total / edge_total, 4) if edge_total else 0.0,
            "curved_ratio":   round(c_total / edge_total, 4) if edge_total else 0.0,
            "note": (
                "straight_ratio > 0.6 → ほぼ直線で構成 (多角形ロゴ等)\n"
                "curved_ratio   > 0.6 → 曲線が主体 (円・有機形状等)"
            ),
        },
    }

    report_path = output_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  レポート    : {report_path}")

    # サマリー
    print(f"\n{sep}")
    print("  解析完了")
    print(f"  直線比率: {report['geometry']['straight_ratio'] * 100:.1f}%")
    print(f"  曲線比率: {report['geometry']['curved_ratio'] * 100:.1f}%")
    print("  色分布:")
    for lbl, info in color_dist.items():
        print(f"    {lbl:12s}: {info['percent']:5.1f}%  "
              f"RGB=({info['rgb'][0]:3d},{info['rgb'][1]:3d},{info['rgb'][2]:3d})")
    print(f"{sep}\n")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analyze_logo",
        description="大型JPEG ロゴの色・形状解析  (読み取り専用)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", help="入力 JPEG パス  (例: input/logo_HUmeta.jpg)")
    p.add_argument(
        "-o", "--output", default=None,
        help="出力ディレクトリ (デフォルト: readonly/output/)",
    )
    p.add_argument(
        "--n-colors", type=int, default=N_COLORS, metavar="N",
        help=f"パレット色数 (デフォルト: {N_COLORS})",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        # input/ ディレクトリを試みる
        project_root = Path(__file__).resolve().parent.parent
        candidate = project_root / "input" / inp.name
        if candidate.exists():
            inp = candidate
        else:
            print(f"エラー: ファイルが見つかりません: {args.input}", file=sys.stderr)
            sys.exit(1)

    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = Path(__file__).resolve().parent / "output"

    global N_COLORS
    N_COLORS = args.n_colors

    process(inp, out_dir)


if __name__ == "__main__":
    main()
