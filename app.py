import gradio as gr
import subprocess
import os
import shutil
import uuid

# StableSRスクリプトのパス
STABLESR_SCRIPT = "scripts/sr_val_ddpm_text_T_vqganfin_old.py"

# チェックポイントファイルのパス (ダウンロード済みを前提)
VQGAN_CKPT = "./vqgan_cfw_00011.ckpt"
STABLESR_CKPT = "./stablesr_turbo.ckpt"

def run_stablesr(input_image_path, ddpm_steps, dec_w, seed, n_samples, colorfix_type):
    # 一時的な出力ディレクトリを作成
    output_dir_name = f"output_{uuid.uuid4()}"
    output_path = os.path.join("outputs", output_dir_name)
    os.makedirs(output_path, exist_ok=True)

    # 入力画像を一時ディレクトリにコピー
    temp_input_dir = os.path.join("temp_inputs", str(uuid.uuid4()))
    os.makedirs(temp_input_dir, exist_ok=True)
    shutil.copy(input_image_path, temp_input_dir)

    # StableSRスクリプトのコマンドを構築
    command = [
        "python", STABLESR_SCRIPT,
        "--config", "configs/stableSRNew/v2-finetune_text_T_512.yaml",
        "--ckpt", STABLESR_CKPT,
        "--init-img", temp_input_dir, # 一時ディレクトリを渡す
        "--outdir", output_path,
        "--ddpm_steps", str(ddpm_steps),
        "--dec_w", str(dec_w),
        "--seed", str(seed),
        "--n_samples", str(n_samples),
        "--vqgan_ckpt", VQGAN_CKPT,
        "--colorfix_type", colorfix_type
    ]

    try:
        # コマンドを実行
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        print("STDOUT:", process.stdout)
        print("STDERR:", process.stderr)

        # 生成された画像ファイルを探す
        generated_images = [os.path.join(output_path, f) for f in os.listdir(output_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if generated_images:
            # 最初の画像を返す (複数生成される可能性もあるが、GradioのImageコンポーネントは単一画像)
            return generated_images[0]
        else:
            return None # 画像が見つからない場合

    except subprocess.CalledProcessError as e:
        print(f"Error running StableSR script: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return None # エラー発生時

    finally:
        # 一時ディレクトリをクリーンアップ (オプション)
        # shutil.rmtree(output_path)
        pass

# Gradioインターフェースの定義
iface = gr.Interface(
    fn=run_stablesr,
    inputs=[
        gr.Image(type="filepath", label="入力画像"),
        gr.Slider(minimum=1, maximum=1000, value=4, step=1, label="DDPM Steps"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Decoder Weight (dec_w)"),
        gr.Number(value=42, label="Seed"),
        gr.Number(value=1, label="Number of Samples (n_samples)"),
        gr.Dropdown(["adain", "wavelet", "nofix"], value="wavelet", label="Color Fix Type")
    ],
    outputs=gr.Image(type="filepath", label="生成された画像"),
    title="StableSR Demo",
    description="入力画像をアップスケールします。"
)

# Gradioアプリの起動
if __name__ == "__main__":
    iface.launch()