import requests
import argparse
from pathlib import Path
import re

API_URL = "http://localhost:8000/qwen_api"  

def send_image(image_path, prompt):
    """Send single image + prompt to your API and return the 'result' field (or error text)."""
    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {"prompt": prompt}
        resp = requests.post(API_URL, files=files, data=data)
    if resp.status_code == 200:
        return resp.json().get("result", "")
    else:
        return f"Error {resp.status_code}: {resp.text}"

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

def clean_init_text(text: str) -> str:
    """Remove common token artifacts and errors from the model output for init.txt.
       Keeps the result short (single line) and strips noisy markers.
    """
    if not text:
        return ""
    # remove explicit error tokens
    text = re.sub(r"ANNOTATION_GENERATION_ERROR", "", text, flags=re.IGNORECASE)
    # remove patterns like (([AA] man)) or ((something))
    text = re.sub(r"\(\([^)]*\)\)", "", text)
    # remove bracketed tokens [[...]] or [...], if present
    text = re.sub(r"\[\[.*?\]\]", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    # collapse whitespace and strip
    text = re.sub(r"\s+", " ", text).strip()
    # if the API returned multiple lines, take first non-empty line
    lines = [ln.strip() for ln in re.split(r"\r?\n", text) if ln.strip()]
    if lines:
        # take the first line, but also avoid grabbing something obviously empty after cleanup
        return lines[0]
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch send images and create init.txt (2-line summary without token)")
    parser.add_argument("--dir", required=True, help="Directory containing images")
    parser.add_argument("--token", required=True, help="Token to include in per-image prompts")
    parser.add_argument("--out", default=None, help="Output directory for .txt files (default: same as --dir)")
    args = parser.parse_args()

    img_dir = Path(args.dir)
    if not img_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.dir}")

    out_dir = Path(args.out) if args.out else img_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in img_dir.iterdir() if p.is_file() and is_image_file(p)])
    if not images:
        print("No images found.")
        raise SystemExit(0)

    # 1) Per-image full descriptions — **these prompts include the token**
    for img_path in images:
        prompt = f"Describe this image in detail. Use this token {args.token}"
        result = send_image(str(img_path), prompt)
        txt_path = out_dir / f"{img_path.stem}.txt"
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(str(result))
        except Exception as exc:
            print(f"Failed to write {txt_path.name}: {exc}")
            continue
        print(f"Wrote: {txt_path.name}")

    # 2) Create single init.txt WITH EXACTLY 2 LINES (no token used in these prompts).
    #    We use the first image as a representative for the init summary. If you want
    #    a true aggregation across all images, see the note below.
    representative_img = str(images[0])

    facial_prompt = (
        "Summarize only the facial features of the person in this image. "
        "Be concise (one short line). Do NOT include any tokens, bracketed markers, or special tags."
    )
    body_prompt = (
        "Summarize only the body features (build, posture, clothing style) of the subject in this image. "
        "Be concise (one short line). Do NOT include any tokens, bracketed markers, or special tags."
    )

    raw_facial = send_image(representative_img, facial_prompt)
    raw_body = send_image(representative_img, body_prompt)

    facial_line = clean_init_text(str(raw_facial))
    body_line = clean_init_text(str(raw_body))

    # Ensure init.txt has exactly two lines (even if empty)
    init_txt_path = out_dir / "init.txt"
    with open(init_txt_path, "w", encoding="utf-8") as init_file:
        init_file.write(f"{facial_line}\n{body_line}\n")

    print(f"Wrote init.txt with 2 lines to: {init_txt_path}")
