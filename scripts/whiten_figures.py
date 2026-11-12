#!/usr/bin/env python3
"""
whiten_figures.py

Replaces the off-white outer background and large pastel container panel fills
with pure white (#FFFFFF) in asr_limitations.png and pipeline_schematic.png,
while cleaning and applying lightened, crisp pastel colors to the inner boxes
and preserving all dark text labels, arrows, outlines, and inner dialogue cards intact.
"""

from pathlib import Path
import shutil
import sys
import argparse
import numpy as np
from PIL import Image


def backup_and_whiten_asr(fig_dir: Path) -> None:
    src_path = fig_dir / "asr_limitations.png"
    bak_path = fig_dir / "asr_limitations.png.bak"
    white_bg_copy = fig_dir / "asr_limitations_white_bg.png"

    # Ensure clean backup exists
    if not bak_path.exists():
        backup_src = Path("metrics_output/backup/asr_limitations.png")
        if backup_src.exists():
            shutil.copy2(backup_src, bak_path)
            print(f"✓ Restored clean backup from {backup_src} to {bak_path}")
        else:
            shutil.copy2(src_path, bak_path)
            print(f"✓ Created backup: {bak_path}")
    else:
        print(f"ℹ Using existing backup: {bak_path}")

    img = Image.open(bak_path).convert("RGB")
    arr = np.array(img)
    r = arr[:, :, 0].astype(int)
    g = arr[:, :, 1].astype(int)
    b = arr[:, :, 2].astype(int)

    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    chroma = mx - mn
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    arr_out = arr.copy()

    # 1. Outer canvas off-white (R, G, B >= 240, neutral chroma)
    mask_outer = (r >= 240) & (g >= 240) & (b >= 240) & (chroma <= 15)
    arr_out[mask_outer] = [255, 255, 255]

    # 2. Left container panel background (light peach fill, x: 21..824, y: 82..931)
    mask_left_bg = np.zeros(arr.shape[:2], dtype=bool)
    mask_left_bg[83:930, 22:823] = (chroma[83:930, 22:823] <= 20) & (lum[83:930, 22:823] >= 235)
    arr_out[mask_left_bg] = [255, 255, 255]

    # Clean & lighten left boxes (peach/coral theme)
    mask_left_box = np.zeros(arr.shape[:2], dtype=bool)
    mask_left_box[83:930, 22:823] = (
        (chroma[83:930, 22:823] >= 22)
        & (lum[83:930, 22:823] >= 190)
        & (r[83:930, 22:823] > g[83:930, 22:823])
    )
    arr_out[mask_left_box] = [254, 228, 222]

    # 3. Right container panel background (light green fill, x: 854..1657, y: 83..931)
    mask_right_bg = np.zeros(arr.shape[:2], dtype=bool)
    mask_right_bg[83:930, 855:1656] = (
        (r[83:930, 855:1656] >= 243)
        & (g[83:930, 855:1656] >= 248)
        & (b[83:930, 855:1656] >= 239)
    )
    arr_out[mask_right_bg] = [255, 255, 255]

    # Clean & lighten right boxes (mint/sage green theme)
    mask_right_box = np.zeros(arr.shape[:2], dtype=bool)
    mask_right_box[83:930, 855:1656] = (
        (b[83:930, 855:1656] <= 236)
        & (g[83:930, 855:1656] >= 240)
        & (r[83:930, 855:1656] <= 242)
        & (lum[83:930, 855:1656] >= 190)
    )
    arr_out[mask_right_box] = [226, 244, 222]

    # Save to both file names
    Image.fromarray(arr_out).save(src_path)
    Image.fromarray(arr_out).save(white_bg_copy)
    print(
        f"✓ Overwrote {src_path.name} and {white_bg_copy.name}:\n"
        f"   - Whitened background: {(mask_outer | mask_left_bg | mask_right_bg).sum():,} px\n"
        f"   - Lightened & cleaned peach boxes: {mask_left_box.sum():,} px\n"
        f"   - Lightened & cleaned green boxes: {mask_right_box.sum():,} px"
    )


def backup_and_whiten_pipeline(fig_dir: Path) -> None:
    src_path = fig_dir / "pipeline_schematic.png"
    bak_path = fig_dir / "pipeline_schematic.png.bak"
    white_bg_copy = fig_dir / "pipeline_schematic_white_bg.png"

    if not bak_path.exists():
        shutil.copy2(src_path, bak_path)
        print(f"✓ Created backup: {bak_path}")
    else:
        print(f"ℹ Using existing backup: {bak_path}")

    img = Image.open(bak_path).convert("RGB")
    arr = np.array(img)
    r = arr[:, :, 0].astype(int)
    g = arr[:, :, 1].astype(int)
    b = arr[:, :, 2].astype(int)

    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    chroma = mx - mn
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    arr_out = arr.copy()

    # 1. Outer canvas off-white (R, G, B >= 240, neutral chroma)
    mask_outer = (r >= 240) & (g >= 240) & (b >= 240) & (chroma <= 15)
    arr_out[mask_outer] = [255, 255, 255]

    # 2. Phase 1 container fill (light peach)
    mask_p1 = (
        (r >= 238)
        & (r <= 255)
        & (g >= 212)
        & (g <= 234)
        & (b >= 206)
        & (b <= 228)
        & (r > g)
        & (g >= b)
    )
    arr_out[mask_p1] = [255, 255, 255]

    # 3. Phase 2 container fill (light cream/yellow)
    mask_p2 = (
        (r >= 246)
        & (r <= 255)
        & (g >= 228)
        & (g <= 240)
        & (b >= 195)
        & (b <= 214)
        & (r > g)
        & (g > b)
    )
    arr_out[mask_p2] = [255, 255, 255]

    # 4. Phase 3 container fill (light blue)
    mask_p3 = (
        (b >= 236)
        & (b <= 255)
        & (g >= 228)
        & (g <= 244)
        & (r >= 216)
        & (r <= 234)
        & (b >= g)
        & (g >= r)
    )
    arr_out[mask_p3] = [255, 255, 255]

    # 5. Bottom panels container backgrounds
    mask_bot_peach = (
        (r >= 230)
        & (r <= 255)
        & (g >= 208)
        & (g <= 234)
        & (b >= 198)
        & (b <= 220)
        & (r > g)
        & (g >= b)
    )
    mask_bot_green = (
        (g >= 232)
        & (g <= 246)
        & (r >= 222)
        & (r <= 240)
        & (b >= 218)
        & (b <= 235)
        & (g >= r)
        & (g >= b)
    )
    arr_out[mask_bot_peach] = [255, 255, 255]
    arr_out[mask_bot_green] = [255, 255, 255]

    # 6. Clean & lighten prompt boxes (WalledEval, XSTest, CoSafe: pastel terracotta)
    mask_prompts = (
        (r >= 215) & (r <= 240) & (g >= 120) & (g <= 160) & (b >= 100) & (b <= 135) & (r - g > 60)
    )
    arr_out[mask_prompts] = [252, 218, 208]

    # 7. Clean & lighten Phase 3 evaluation boxes (pastel sky blue)
    mask_blue_boxes = (
        (b >= 215) & (b <= 242) & (g >= 195) & (g <= 225) & (r >= 180) & (r <= 215) & (b >= g) & (g >= r)
    )
    arr_out[mask_blue_boxes] = [220, 234, 250]

    # 8. Clean & lighten Bottom Right summary box (pastel mint/sage green)
    mask_green_bot = (
        (g >= 195)
        & (g <= 218)
        & (r >= 175)
        & (r <= 200)
        & (b >= 175)
        & (b <= 200)
        & (g - r > 10)
        & (g - b > 10)
    )
    arr_out[mask_green_bot] = [226, 244, 222]

    # 9. Clean & lighten header banners
    mask_h1 = (r >= 215) & (r <= 235) & (g >= 180) & (g <= 200) & (b >= 170) & (b <= 190)
    arr_out[mask_h1] = [252, 228, 222]

    mask_h2 = (r >= 230) & (r <= 248) & (g >= 195) & (g <= 215) & (b >= 135) & (b <= 155)
    arr_out[mask_h2] = [254, 236, 196]

    mask_h3 = (r >= 190) & (r <= 210) & (g >= 205) & (g <= 225) & (b >= 225) & (b <= 245)
    arr_out[mask_h3] = [218, 232, 248]

    # 10. Clean & lighten Phase 2 Target Model container box
    mask_model = (r >= 246) & (r <= 255) & (g >= 240) & (g <= 252) & (b >= 198) & (b <= 214)
    arr_out[mask_model] = [254, 244, 212]

    # Clean & lighten Model sub-boxes (Gemini, Llama, Mistral)
    mask_submodel = (r >= 225) & (r <= 245) & (g >= 185) & (g <= 205) & (b >= 125) & (b <= 145)
    arr_out[mask_submodel] = [254, 232, 182]

    # 11. Clean & lighten Defense boxes (Baseline, SafeDecoding, LlamaGuard-3)
    mask_defenses = (
        (r >= 195) & (r <= 235) & (g >= 175) & (g <= 205) & (b >= 140) & (b <= 170) & (r >= g) & (g > b)
    )
    arr_out[mask_defenses] = [252, 230, 200]

    # 12. Clean & lighten dataset sub-category tags in Phase 1 (sage)
    mask_tags = (g >= 180) & (g <= 200) & (r >= 160) & (r <= 185) & (b >= 145) & (b <= 170) & (g > r)
    arr_out[mask_tags] = [226, 240, 222]

    # 13. Clean & lighten pipeline flow labels / sand boxes
    mask_sand = (
        (r >= 200) & (r <= 220) & (g >= 175) & (g <= 195) & (b >= 165) & (b <= 185) & (r > g) & (g >= b)
    )
    arr_out[mask_sand] = [246, 232, 226]

    # Save to both file names
    Image.fromarray(arr_out).save(src_path)
    Image.fromarray(arr_out).save(white_bg_copy)
    print(
        f"✓ Overwrote {src_path.name} and {white_bg_copy.name}:\n"
        f"   - Whitened background: {(mask_outer | mask_p1 | mask_p2 | mask_p3 | mask_bot_peach | mask_bot_green).sum():,} px\n"
        f"   - Lightened & cleaned prompt boxes: {mask_prompts.sum():,} px\n"
        f"   - Lightened & cleaned blue boxes: {mask_blue_boxes.sum():,} px\n"
        f"   - Lightened & cleaned model boxes: {(mask_model | mask_submodel).sum():,} px\n"
        f"   - Lightened & cleaned defense boxes: {mask_defenses.sum():,} px\n"
        f"   - Lightened & cleaned green summary: {mask_green_bot.sum():,} px"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Whiten figures background and container panels with clean lightened box fills."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        help="Target figure to process: 'asr', 'pipeline', 'all', or file path.",
    )
    parser.add_argument(
        "--fig-dir",
        default="metrics_output/figures",
        help="Figures directory (default: metrics_output/figures).",
    )
    args = parser.parse_args()

    fig_dir = Path(args.fig_dir)
    print(f"Target directory: {fig_dir.resolve()}")

    target = args.target.lower()
    run_asr = "asr" in target or target == "all"
    run_pipeline = "pipeline" in target or target == "all"

    if run_asr:
        backup_and_whiten_asr(fig_dir)
    if run_pipeline:
        backup_and_whiten_pipeline(fig_dir)

    print("\n✓ Whiten and recolor processing complete.")


if __name__ == "__main__":
    main()
