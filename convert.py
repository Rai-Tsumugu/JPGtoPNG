#!/usr/bin/env python3
"""
JPG → SVG → PNG  色忠実変換パイプライン
==========================================

4 段階パイプライン:

  [1] 読み込み & ノイズ除去
        双方向フィルタ + メジアンフィルタで JPEG アーティファクトを除去

  [2] 色量子化
        CIE Lab 空間で MiniBatch K-means を実行。
        ジャギーによる中間色を排除し、使用色を確定する。
        近似色クラスタは CIE76 距離で自動統合。

  [3] ベクター化 → SVG
        量子化済み画像をトレースして SVG を生成。
        バックエンド優先: vtracer（Rust 製・高速・高品質）
                    → パイプライン自前実装（輪郭抽出 + Catmull-Rom Bezier）

  [4] ラスタライズ → PNG
        SVG を元解像度で PNG に変換。
        バックエンド優先: cairosvg → svglib+reportlab → Inkscape → ImageMagick

JPG（量子化済み）・SVG・PNG はすべて同一カラーパレットを共有します。

使い方:
  python convert.py logo.jpg              # output/ に jpg + svg + png を全出力
  python convert.py logo.jpg -k 16        # 色数を絞る
  python convert.py logo.jpg --jpg-only   # 量子化 JPG のみ（色確認用）
  python convert.py logo.jpg --svg-only   # SVG のみ
  python convert.py logo.jpg --png-only   # PNG のみ（SVG は一時ファイル）
  python convert.py logo.jpg -o out/      # 出力ディレクトリ指定
  python convert.py logo.jpg -q           # サイレント実行
"""

import argparse
import io
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent
_INPUT_DIR = _ROOT / "input"
_OUTPUT_DIR = _ROOT / "output"

sys.path.insert(0, str(_ROOT))

from utils.image_utils import load_image, preprocess_jpeg
from stages.clustering import cluster_colors, merge_similar_clusters, quantized_image
from stages.masking import generate_masks
from stages.contour_extractor import extract_all_contours
from stages.vectorizer import contours_to_svg
from stages.renderer import render_direct, save_png


# ─────────────────────────────────────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────

def _detect_bg_color(q_img: np.ndarray) -> Tuple[int, int, int]:
    """量子化画像の四隅からサンプリングして背景色を推定する。"""
    h, w = q_img.shape[:2]
    corners = [
        tuple(q_img[0, 0].tolist()),
        tuple(q_img[0, w - 1].tolist()),
        tuple(q_img[h - 1, 0].tolist()),
        tuple(q_img[h - 1, w - 1].tolist()),
    ]
    return Counter(corners).most_common(1)[0][0]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 バックエンド: JPG → SVG
# ─────────────────────────────────────────────────────────────────────────────

def _svg_via_vtracer(png_path: str, svg_path: str) -> bool:
    """vtracer でカラートレース。成功すれば True を返す。"""
    try:
        import vtracer
    except ImportError:
        return False
    try:
        vtracer.convert_image_to_svg_py(
            png_path, svg_path,
            colormode="color",
            hierarchical="stacked",
            mode="spline",
            filter_speckle=4,
            color_precision=6,
            layer_difference=16,
            corner_threshold=60,
            length_threshold=4.0,
            max_iterations=10,
            splice_threshold=45,
            path_precision=3,
        )
        return True
    except Exception as exc:
        print(f"    [!] vtracer: {exc}")
        return False


def _svg_via_pipeline(
    q_img: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    svg_path: str,
    epsilon_ratio: float,
    use_curves: bool,
    morph_kernel: int,
    bg_color: Tuple[int, int, int],
    verbose: bool,
) -> List:
    """フォールバック: 色領域ごとに輪郭抽出 → Catmull-Rom Bezier → SVG
    contour_data を返す（PNG 直接レンダリングのフォールバック用）。"""
    h, w = q_img.shape[:2]
    masks = generate_masks(labels, centers, morph_kernel_size=morph_kernel)
    contour_data = extract_all_contours(masks, image_area=h * w, epsilon_ratio=epsilon_ratio)
    if verbose:
        n_contours = sum(len(c[0]) for c in contour_data)
        print(f"    {len(contour_data)} 色領域 / {n_contours} 輪郭")
    contours_to_svg(
        contour_data, width=w, height=h,
        output_path=svg_path,
        use_curves=use_curves,
        background_color=bg_color,
    )
    return contour_data


def jpg_to_svg(
    q_img: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    svg_path: str,
    epsilon_ratio: float = 0.002,
    use_curves: bool = True,
    morph_kernel: int = 3,
    verbose: bool = True,
) -> Tuple[str, Optional[List]]:
    """量子化画像を SVG に変換する。
    Returns: (backend_name, contour_data | None)
      contour_data は pipeline バックエンド使用時のみ返す（PNG 直描画フォールバック用）。
    """
    bg_color = _detect_bg_color(q_img)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tmp_png = tf.name
    Image.fromarray(q_img).save(tmp_png)

    try:
        if verbose:
            print("    vtracer を試みます …")
        if _svg_via_vtracer(tmp_png, svg_path):
            return "vtracer", None

        if verbose:
            print("    -> 利用不可。パイプライン自前ベクタライザーを使用 …")
        contour_data = _svg_via_pipeline(
            q_img, labels, centers, svg_path,
            epsilon_ratio=epsilon_ratio,
            use_curves=use_curves,
            morph_kernel=morph_kernel,
            bg_color=bg_color,
            verbose=verbose,
        )
        return "pipeline", contour_data
    finally:
        Path(tmp_png).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 バックエンド: SVG → PNG
# ─────────────────────────────────────────────────────────────────────────────

def _png_via_cairosvg(svg_path: str, w: int, h: int) -> Optional[np.ndarray]:
    try:
        import cairosvg
        data = cairosvg.svg2png(
            url=str(Path(svg_path).resolve()), output_width=w, output_height=h,
        )
        return np.array(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)
    except ImportError:
        return None
    except Exception as exc:
        print(f"    cairosvg: {exc}")
        return None


def _png_via_svglib(svg_path: str, w: int, h: int) -> Optional[np.ndarray]:
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        d = svg2rlg(svg_path)
        if d is None:
            return None
        d.transform = (w / d.width if d.width else 1, 0, 0, h / d.height if d.height else 1, 0, 0)
        d.width, d.height = w, h
        data = renderPM.drawToString(d, fmt="PNG")
        return np.array(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)
    except ImportError:
        return None
    except Exception as exc:
        print(f"    svglib: {exc}")
        return None


def _png_via_inkscape(svg_path: str, out: str, w: int, h: int) -> Optional[np.ndarray]:
    try:
        r = subprocess.run(
            ["inkscape", svg_path,
             f"--export-filename={out}",
             f"--export-width={w}", f"--export-height={h}"],
            capture_output=True, timeout=120,
        )
        if r.returncode == 0 and Path(out).exists():
            return np.array(Image.open(out).convert("RGB"), dtype=np.uint8)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _png_via_imagemagick(svg_path: str, out: str, w: int, h: int) -> Optional[np.ndarray]:
    try:
        if subprocess.run(["magick", "--version"], capture_output=True, timeout=10).returncode != 0:
            return None
        r = subprocess.run(
            ["magick", "-density", "150", "-resize", f"{w}x{h}!", svg_path, out],
            capture_output=True, timeout=120,
        )
        if r.returncode == 0 and Path(out).exists():
            return np.array(Image.open(out).convert("RGB"), dtype=np.uint8)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def svg_to_png(
    svg_path: str,
    png_path: str,
    w: int,
    h: int,
    contour_data: Optional[List] = None,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    verbose: bool = True,
) -> str:
    """SVG を PNG にラスタライズする。使用したバックエンド名を返す。
    contour_data が渡された場合は OpenCV 直描画を最終フォールバックとして使用。"""
    backends = [
        ("cairosvg",         lambda: _png_via_cairosvg(svg_path, w, h)),
        ("svglib+reportlab", lambda: _png_via_svglib(svg_path, w, h)),
        ("inkscape",         lambda: _png_via_inkscape(svg_path, png_path, w, h)),
        ("imagemagick",      lambda: _png_via_imagemagick(svg_path, png_path, w, h)),
    ]

    for name, fn in backends:
        if verbose:
            print(f"    {name} を試みます …")
        arr = fn()
        if arr is not None:
            if not Path(png_path).exists():
                Image.fromarray(arr.astype(np.uint8)).save(png_path, "PNG", optimize=True)
            kb = Path(png_path).stat().st_size / 1024
            if verbose:
                print(f"    -> {png_path}  ({kb:.1f} KB)")
            return name
        if verbose:
            print("    -> スキップ")

    # 追加依存なしのフォールバック: OpenCV fillPoly 直接レンダリング
    if contour_data is not None:
        if verbose:
            print("    OpenCV 直接レンダリング（追加依存なし）…")
        arr = render_direct(contour_data, width=w, height=h, background=bg_color)
        save_png(arr, png_path)
        kb = Path(png_path).stat().st_size / 1024
        if verbose:
            print(f"    -> {png_path}  ({kb:.1f} KB)")
        return "opencv-direct"

    print(
        "\n  [!] SVG→PNG バックエンドが見つかりません。以下のいずれかをインストールしてください:\n"
        "      pip install cairosvg\n"
        "      pip install svglib reportlab\n"
        "      Inkscape を PATH に追加\n"
        "      ImageMagick (magick v7+) を PATH に追加",
        file=sys.stderr,
    )
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# メインパイプライン
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_input(name: str) -> Path:
    p = Path(name)
    if p.exists():
        return p
    c = _INPUT_DIR / p.name
    return c if c.exists() else p


def run(
    input_path: str,
    out_dir: Path,
    n_clusters: int = 24,
    merge_threshold: float = 12.0,
    denoise: int = 5,
    epsilon_ratio: float = 0.002,
    morph_kernel: int = 3,
    use_curves: bool = True,
    emit_jpg: bool = True,
    emit_svg: bool = True,
    emit_png: bool = True,
    verbose: bool = True,
) -> None:
    sep = "=" * 54
    if verbose:
        print(f"\n{sep}")
        print("  JPG → SVG → PNG  (色忠実変換)")
        print(f"  入力: {Path(input_path).resolve()}")
        print(f"  出力: {out_dir.resolve()}/")
        print(f"{sep}")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem

    # ── [1] 読み込み & ノイズ除去 ──────────────────────────────────────────────
    if verbose:
        print("\n[1/4] 読み込み & ノイズ除去")
    original = load_image(input_path)
    h, w = original.shape[:2]
    if verbose:
        print(f"      {w}x{h}px")
    image = preprocess_jpeg(original, denoise_strength=denoise) if denoise > 0 else original

    # ── [2] 色量子化 ──────────────────────────────────────────────────────────
    if verbose:
        print(f"\n[2/4] 色量子化  k={n_clusters}  merge-threshold={merge_threshold}")
    labels, centers = cluster_colors(image, n_clusters=n_clusters)
    if merge_threshold > 0:
        labels, centers = merge_similar_clusters(labels, centers, merge_threshold)
    q_img = quantized_image(labels, centers)
    if verbose:
        print(f"      {len(centers)} 色確定（ジャギー中間色を除去）")

    if emit_jpg:
        jpg_path = str(out_dir / f"{stem}.jpg")
        Image.fromarray(q_img).save(jpg_path, "JPEG", quality=95)
        if verbose:
            kb = Path(jpg_path).stat().st_size / 1024
            print(f"      -> {jpg_path}  ({kb:.1f} KB)")

    if not emit_svg and not emit_png:
        _print_done(verbose, sep)
        return

    # ── [3] ベクター化 → SVG ──────────────────────────────────────────────────
    if verbose:
        print("\n[3/4] ベクター化 → SVG")

    svg_tmp_path: Optional[str] = None
    if emit_svg:
        svg_path = str(out_dir / f"{stem}.svg")
    else:
        _t = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
        svg_path = _t.name
        svg_tmp_path = _t.name
        _t.close()

    backend, contour_data = jpg_to_svg(
        q_img, labels, centers, svg_path,
        epsilon_ratio=epsilon_ratio,
        use_curves=use_curves,
        morph_kernel=morph_kernel,
        verbose=verbose,
    )
    if verbose:
        kb = Path(svg_path).stat().st_size / 1024
        label = svg_path if emit_svg else "(一時ファイル)"
        print(f"      -> {label}  ({kb:.1f} KB)  [{backend}]")

    # ── [4] ラスタライズ → PNG ─────────────────────────────────────────────────
    if emit_png:
        if verbose:
            print("\n[4/4] ラスタライズ → PNG")
        png_path = str(out_dir / f"{stem}.png")
        bg_color = _detect_bg_color(q_img)
        png_backend = svg_to_png(
            svg_path, png_path, w, h,
            contour_data=contour_data,
            bg_color=bg_color,
            verbose=verbose,
        )
        if verbose:
            print(f"      [{png_backend}]")

    if svg_tmp_path:
        Path(svg_tmp_path).unlink(missing_ok=True)

    _print_done(verbose, sep)


def _print_done(verbose: bool, sep: str) -> None:
    if verbose:
        print(f"\n{sep}\n  完了\n{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        prog="convert",
        description="JPG → SVG → PNG  色忠実変換パイプライン",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "input",
        help="入力 JPEG（ファイル名のみの場合は input/ を自動検索）",
    )
    p.add_argument(
        "-o", "--output", default=None,
        help="出力ディレクトリ（デフォルト: output/）",
    )

    gm = p.add_argument_group("出力モード（デフォルト: jpg + svg + png を全出力）")
    ex = gm.add_mutually_exclusive_group()
    ex.add_argument("--jpg-only", action="store_true",
                    help="量子化 JPG のみ出力（色確認用）")
    ex.add_argument("--svg-only", action="store_true",
                    help="SVG のみ出力")
    ex.add_argument("--png-only", action="store_true",
                    help="PNG のみ出力（SVG は一時ファイルとして処理後削除）")

    gc = p.add_argument_group("色量子化オプション")
    gc.add_argument(
        "-k", "--clusters", type=int, default=24, metavar="N",
        help="初期色クラスタ数（デフォルト: 24）",
    )
    gc.add_argument(
        "--merge-threshold", type=float, default=12.0, metavar="D",
        help="類似クラスタ統合の CIE76 距離閾値（デフォルト: 12.0）",
    )
    gc.add_argument(
        "--denoise", type=int, default=5, metavar="D",
        help="JPEG ノイズ除去強度 0–15（0=無効、デフォルト: 5）",
    )

    gv = p.add_argument_group("ベクター化オプション（vtracer 非使用時のみ有効）")
    gv.add_argument(
        "--epsilon", type=float, default=0.002, metavar="E",
        help="RDP 簡略化比率（デフォルト: 0.002）",
    )
    gv.add_argument(
        "--no-curves", action="store_true",
        help="Bezier 曲線を使わずポリラインで出力",
    )
    gv.add_argument(
        "--morph-kernel", type=int, default=3, metavar="K",
        help="形態学カーネルサイズ（デフォルト: 3）",
    )

    p.add_argument("-q", "--quiet", action="store_true", help="進捗出力を抑制")

    args = p.parse_args()

    inp = _resolve_input(args.input)
    if not inp.exists():
        print(f"Error: ファイルが見つかりません: {args.input}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output) if args.output else _OUTPUT_DIR

    emit_jpg = not (args.svg_only or args.png_only)
    emit_svg = not (args.jpg_only or args.png_only)
    emit_png = not (args.jpg_only or args.svg_only)

    run(
        str(inp), out_dir,
        n_clusters=args.clusters,
        merge_threshold=args.merge_threshold,
        denoise=args.denoise,
        epsilon_ratio=args.epsilon,
        morph_kernel=args.morph_kernel,
        use_curves=not args.no_curves,
        emit_jpg=emit_jpg,
        emit_svg=emit_svg,
        emit_png=emit_png,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
