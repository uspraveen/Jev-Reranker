"""Render the Jev-Reranker overview video from docs/media/video.html.

Frames are screenshots of a deterministic time-driven page (t = seconds),
encoded at 30 fps with ffmpeg. Usage:

    python docs/media/render_video.py probe      # 4 sample frames
    python docs/media/render_video.py full       # full render + encode
"""

import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
HTML = (HERE / "video.html").resolve()
FRAMES = Path(r"C:\Users\Praveen Raj U S\AppData\Local\Temp\jev_frames")
OUT = HERE / "jev_reranker_overview.mp4"
FPS = 30
DURATION = 38.6
N = int(DURATION * FPS)


def shoot(page, t: float, path: Path) -> None:
    page.evaluate(f"location.hash = '#'; render({t});")
    # evaluate only sets the hash; render() is called directly in the page context
    page.screenshot(path=str(path))


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    FRAMES.mkdir(parents=True, exist_ok=True)
    frames = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", args=["--force-device-scale-factor=1"])
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(HTML.as_uri())
        page.evaluate("render(0)")

        if mode == "probe":
            for t in (1.8, 8.6, 13.2, 22.6, 29.8, 36.2):
                page.evaluate(f"render({t})")
                path = HERE / f"_probe_{t}.png"
                page.screenshot(path=str(path))
                frames.append(path)
                print("probe", t)
        else:
            t0 = time.time()
            for i in range(N):
                page.evaluate(f"render({i / FPS})")
                page.screenshot(path=str(FRAMES / f"{i:05d}.png"))
                if i % 150 == 0:
                    print(f"frame {i}/{N} ({time.time() - t0:.0f}s)", flush=True)
            print(f"rendered {N} frames in {time.time() - t0:.0f}s")

        browser.close()

    if mode != "probe":
        ffmpeg = subprocess.run(
            [sys.executable, "-c",
             "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"],
            capture_output=True, text=True, check=True).stdout.strip()
        subprocess.run([
            ffmpeg, "-y", "-framerate", str(FPS),
            "-i", str(FRAMES / "%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            "-movflags", "+faststart", str(OUT),
        ], check=True)
        print("encoded", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
