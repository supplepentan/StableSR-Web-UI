import gradio as gr
import os
import tempfile
import time
import PIL
import numpy as np
import copy
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import trange
from itertools import islice
from einops import rearrange, repeat
from torch import autocast
from pytorch_lightning import seed_everything
import torch.nn.functional as F

from ldm.util import instantiate_from_config
from scripts.wavelet_color_fix import (
    wavelet_reconstruction,
    adaptive_instance_normalization,
)

# Import necessary functions from predict.py
# Since we can't directly import from a file as a module without modifying sys.path,
# we'll copy the necessary helper functions and the Predictor class logic here.
# In a real-world scenario, you might refactor predict.py into a proper package
# or import it dynamically if it's guaranteed to be in the Python path.


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]  # [250,]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def load_img(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {image_path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


# Initialize models globally to avoid reloading on each prediction

# --- モデルファイルのパスを指定 ---
MODEL_DIR = "models"
STABLESR_CKPT_PATH = os.path.join(MODEL_DIR, "stablesr_turbo.ckpt")
VQGAN_CKPT_PATH = os.path.join(MODEL_DIR, "vqgan_cfw_00011.ckpt")
# --------------------------------

# --- モデルファイルの存在チェック ---
if not os.path.exists(STABLESR_CKPT_PATH) or not os.path.exists(VQGAN_CKPT_PATH):
    print("=" * 50)
    print("エラー: モデルファイルが見つかりません。")
    print(
        f"'{STABLESR_CKPT_PATH}' と '{VQGAN_CKPT_PATH}' が存在することを確認してください。"
    )
    print("`download_models.py` を実行してモデルをダウンロードしてください。")
    print("=" * 50)
    exit()

print("Loading StableSR models...")
config = OmegaConf.load("configs/stableSRNew/v2-finetune_text_T_512.yaml")
model = load_model_from_config(config, STABLESR_CKPT_PATH)
device = torch.device("cuda")
model.configs = config
model = model.to(device)

vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
vq_model = load_model_from_config(vqgan_config, VQGAN_CKPT_PATH)
vq_model = vq_model.to(device)
print("Models loaded.")


def predict_sr(
    input_image: Image.Image,
    ddpm_steps: int,
    fidelity_weight: float,
    upscale: float,
    tile_overlap: int,
    colorfix_type: str,
    seed: int,
):
    if seed is None:
        seed = int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {seed}")

    vq_model.decoder.fusion_w = fidelity_weight

    seed_everything(seed)

    n_samples = 1
    device = torch.device("cuda")

    # Use a temporary file to safely handle the input image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_input_path = temp_file.name
        input_image.save(temp_input_path)

    try:
        cur_image = load_img(temp_input_path).to(device)
    finally:
        # Clean up the temporary input file
        os.remove(temp_input_path)

    cur_image = F.interpolate(
        cur_image,
        size=(int(cur_image.size(-2) * upscale), int(cur_image.size(-1) * upscale)),
        mode="bicubic",
    )

    model.register_schedule(
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=0.00085,
        linear_end=0.0120,
        cosine_s=8e-3,
    )
    model.num_timesteps = 1000

    sqrt_alphas_cumprod = copy.deepcopy(model.sqrt_alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = copy.deepcopy(model.sqrt_one_minus_alphas_cumprod)

    use_timesteps = set(space_timesteps(1000, [ddpm_steps]))
    last_alpha_cumprod = 1.0
    new_betas = []
    timestep_map = []
    for i, alpha_cumprod in enumerate(model.alphas_cumprod):
        if i in use_timesteps:
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
            timestep_map.append(i)
    new_betas = [beta.data.cpu().numpy() for beta in new_betas]
    model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
    model.num_timesteps = 1000
    model.ori_timesteps = list(use_timesteps)
    model.ori_timesteps.sort()
    model.to(device)  # Ensure model is on the correct device

    precision_scope = autocast
    input_size = 512

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                init_image = cur_image
                init_image = init_image.clamp(-1.0, 1.0)
                ori_size = None

                print(init_image.size())

                if init_image.size(-1) < input_size or init_image.size(-2) < input_size:
                    ori_size = init_image.size()
                    new_h = max(ori_size[-2], input_size)
                    new_w = max(ori_size[-1], input_size)
                    init_template = torch.zeros(1, init_image.size(1), new_h, new_w).to(
                        init_image.device
                    )
                    init_template[:, :, : ori_size[-2], : ori_size[-1]] = init_image
                else:
                    init_template = init_image

                init_latent = model.get_first_stage_encoding(
                    model.encode_first_stage(init_template)
                )  # move to latent space
                text_init = [""] * n_samples
                semantic_c = model.cond_stage_model(text_init)

                noise = torch.randn_like(init_latent)
                t = repeat(torch.tensor([999]), "1 -> b", b=init_image.size(0))
                t = t.to(device).long()
                x_T = model.q_sample_respace(
                    x_start=init_latent,
                    t=t,
                    sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                    sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                    noise=noise,
                )
                samples, _ = model.sample_canvas(
                    cond=semantic_c,
                    struct_cond=init_latent,
                    batch_size=init_image.size(0),
                    timesteps=ddpm_steps,
                    time_replace=ddpm_steps,
                    x_T=x_T,
                    return_intermediates=True,
                    tile_size=int(input_size / 8),
                    tile_overlap=tile_overlap,
                    batch_size_sample=n_samples,
                )
                _, enc_fea_lq = vq_model.encode(init_template)
                x_samples = vq_model.decode(
                    samples * 1.0 / model.scale_factor, enc_fea_lq
                )
                if ori_size is not None:
                    x_samples = x_samples[:, :, : ori_size[-2], : ori_size[-1]]
                if colorfix_type == "adain":
                    x_samples = adaptive_instance_normalization(x_samples, init_image)
                elif colorfix_type == "wavelet":
                    x_samples = wavelet_reconstruction(x_samples, init_image)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                output_image_np = (
                    255.0 * rearrange(x_samples[0].cpu().numpy(), "c h w -> h w c")
                ).astype(np.uint8)
                output_image = Image.fromarray(output_image_np)

    # --- 追加: 生成された画像を保存 ---
    # 出力ディレクトリを作成
    output_dir = "outputs/webui"
    os.makedirs(output_dir, exist_ok=True)

    # タイムスタンプとシード値を使ってユニークなファイル名を生成
    timestamp = int(time.time())
    output_filename = f"{timestamp}_seed{seed}.png"
    output_path = os.path.join(output_dir, output_filename)
    output_image.save(output_path)
    print(f"Output image saved to: {output_path}")

    return output_image


# Gradio Interface
iface = gr.Interface(
    fn=predict_sr,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Slider(minimum=1, maximum=40, value=4, label="DDPM Steps"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.5, label="Fidelity Weight"),
        gr.Slider(
            minimum=1.0, maximum=8.0, value=4.0, step=0.5, label="Upscale Factor"
        ),
        gr.Slider(minimum=0, maximum=64, value=32, step=1, label="Tile Overlap"),
        gr.Radio(["adain", "wavelet", "none"], value="adain", label="Color Fix Type"),
        gr.Number(label="Seed (leave blank for random)", precision=0),
    ],
    outputs=gr.Image(type="pil", label="Output Image"),
    title="StableSR Gradio Demo",
    description="Super-resolution with StableSR model.",
    allow_flagging="never",
)

if __name__ == "__main__":
    iface.launch()
