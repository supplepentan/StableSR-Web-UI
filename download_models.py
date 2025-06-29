import requests
import os
from tqdm import tqdm

# --- ここに保存先ディレクトリを指定 ---
# 例: "models" や "checkpoints" など
DOWNLOAD_DIR = "models"
# ------------------------------------

urls = {
    "vqgan_cfw_00011.ckpt": "https://huggingface.co/Iceclear/StableSR/resolve/main/vqgan_cfw_00011.ckpt",
    "stablesr_turbo.ckpt": "https://huggingface.co/Iceclear/StableSR/resolve/main/stablesr_turbo.ckpt",
}


def download_file(url, filepath):
    """指定されたURLからファイルをダウンロードし、指定されたパスに保存する"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    with open(filepath, "wb") as file, tqdm(
        desc=os.path.basename(filepath),
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)

    if total_size != 0 and bar.n != total_size:
        print(f"ERROR: Download for {os.path.basename(filepath)} might be incomplete.")


# 保存先ディレクトリが存在しない場合は作成
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
print(f"Models will be saved to '{os.path.abspath(DOWNLOAD_DIR)}' directory.")

for filename, url in urls.items():
    # 保存先の完全なファイルパスを構築
    filepath = os.path.join(DOWNLOAD_DIR, filename)

    if os.path.exists(filepath):
        print(f"'{filepath}' already exists, skipping download.")
    else:
        print(f"Downloading {filename}...")
        download_file(url, filepath)

print("Downloads complete!")
