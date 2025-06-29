<p align="center">
  <img src="https://user-images.githubusercontent.com/22350795/236680126-0b1cdd62-d6fc-4620-b998-75ed6c31bf6f.png" height=40>
</p>

## Exploiting Diffusion Prior for Real-World Image Super-Resolution (実世界画像超解像のための拡散事前知識の活用)

[Paper](https://arxiv.org/abs/2305.07015) | [Project Page](https://iceclear.github.io/projects/stablesr/) | [Video](https://www.youtube.com/watch?v=5MZy9Uhpkw4) | [WebUI](https://github.com/pkuliyi2015/sd-webui-stablesr) | [ModelScope](https://modelscope.cn/models/xhlin129/cv_stablesr_image-super-resolution/summary) | [ComfyUI](https://github.com/gameltb/comfyui-stablesr)

<a href="https://colab.research.google.com/drive/11SE2_oDvbYtcuHDbaLAxsKk_o3flsO1T?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> [![Hugging Face](https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/Iceclear/StableSR) [![Replicate](https://img.shields.io/badge/Demo-%F0%9F%9A%80%20Replicate-blue)](https://replicate.com/cjwbw/stablesr) [![OpenXLab](https://img.shields.io/badge/Demo-%F0%9F%90%BC%20OpenXLab-blue)](https://openxlab.org.cn/apps/detail/Iceclear/StableSR) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=IceClear/StableSR)

[Jianyi Wang](https://iceclear.github.io/), [Zongsheng Yue](https://zsyoaoa.github.io/), [Shangchen Zhou](https://shangchenzhou.com/), [Kelvin C.K. Chan](https://ckkelvinchan.github.io/), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)

S-Lab, Nanyang Technological University

<img src="assets/network.png" width="800px"/>

:star: StableSR があなたの画像やプロジェクトに役立つ場合は、このリポジトリにスターを付けてください。ありがとうございます！ :hugs:

### Demo on real-world SR (実世界 SR のデモ)

[<img src="assets/imgsli_1.jpg" height="223px"/>](https://imgsli.com/MTc2MTI2) [<img src="assets/imgsli_2.jpg" height="223px"/>](https://imgsli.com/MTc2MTE2) [<img src="assets/imgsli_3.jpg" height="223px"/>](https://imgsli.com/MTc2MTIw)
[<img src="assets/imgsli_8.jpg" height="223px"/>](https://imgsli.com/MTc2MjUy) [<img src="assets/imgsli_4.jpg" height="223px"/>](https://imgsli.com/MTc2MTMy) [<img src="assets/imgsli_5.jpg" height="223px"/>](https://imgsli.com/MTc2MTMz)
[<img src="assets/imgsli_9.jpg" height="214px"/>](https://imgsli.com/MTc2MjQ5) [<img src="assets/imgsli_6.jpg" height="214px"/>](https://imgsli.com/MTc2MTM0) [<img src="assets/imgsli_7.jpg" height="214px"/>](https://imgsli.com/MTc2MTM2) [<img src="assets/imgsli_10.jpg" height="214px"/>](https://imgsli.com/MTc2MjU0)

詳細な評価については、[論文](https://arxiv.org/abs/2305.07015)を参照してください。

### Demo on 4K Results (4K 結果のデモ)

- StableSR は理論上、任意のアップスケーリングが可能です。以下は 4K（4096x6144）を超える結果の 4 倍の例です。

[<img src="assets/main-fig.png" width="800px"/>](https://imgsli.com/MjIzMjQx)

```bash
# DDIM w/ negative prompts
python scripts/sr_val_ddim_text_T_negativeprompt_canvas_tile.py --config configs/stableSRNew/v2-finetune_text_T_768v.yaml --ckpt stablesr_768v_000139.ckpt --vqgan_ckpt vqgan_finetune_00011.ckpt --init-img ./inputs/test_example/ --outdir ../output/ --ddim_steps 20 --dec_w 0.0 --colorfix_type wavelet --scale 7.0 --use_negative_prompt --upscale 4 --seed 42 --n_samples 1 --input_size 768 --tile_overlap 48 --ddim_eta 1.0
```

- **その他の例**。
  - [4K Demo1](https://imgsli.com/MTc4MDg3)は、[こちら](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111)の画像の 4 倍 SR です。
  - [4K Demo2](https://imgsli.com/MTc4NDk2)は、[こちら](https://github.com/Mikubill/sd-webui-controlnet/blob/main/tests/images/ski.jpg)の画像の 8 倍 SR です。
  - その他の比較は[こちら](https://github.com/IceClear/StableSR/issues/2)と[こちら](https://github.com/pkuliyi2015/sd-webui-stablesr)で見つけることができます。

### Dependencies and Installation (依存関係とインストール)

- Pytorch == 1.12.1
- CUDA == 11.7
- pytorch-lightning==1.4.2
- xformers == 0.0.16 (Optional)
- Other required packages in `environment.yaml`

#### Local Setup (without Conda) (ローカル環境構築 (Conda なし))

このプロジェクトは、`conda` を使用せずに `venv` と `pip` で環境を構築できます。

1.  **仮想環境の作成と有効化**

    ```bash
    python -m venv venv
    # Windowsの場合
    .\venv\Scripts\activate
    # macOS/Linuxの場合
    source venv/bin/activate
    ```

2.  **必要なライブラリのインストール**

    - **ステップ 1: PyTorch のインストール (CUDA 11.7 対応版)**
      まず、お使いの環境に合った PyTorch をインストールします。
      ```bash
      pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu117
      ```
    - **ステップ 2: CLIP のインストール**
      次に、OpenAI のリポジトリから直接 CLIP をインストールします。
      ```bash
      pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
      ```
    - **ステップ 3: taming-transformers のインストール**
      次に、`taming-transformers` をリポジトリから直接インストールします。
      ```bash
      pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
      ```
    - **ステップ 4: その他の依存関係のインストール**
      最後に、`requirements.txt` ファイルを使って、残りのライブラリをすべてインストールします。
      ```bash
      pip install -r requirements.txt
      ```
    - **ステップ 5: プロジェクト自体のインストール**
      最後に、このプロジェクト自体を Python から利用できるようにインストールします。
      ```bash
      pip install -e .
      ```

#### モデルのダウンロード

環境構築が完了したら、次に StableSR のモデルファイルをダウンロードします。プロジェクトのルートディレクトリで以下のコマンドを実行すると、必要なモデルが `models` ディレクトリにダウンロードされます。

```bash
python download_models.py
```

#### Gradio Web UI での実行 (推奨)

このリポジトリには、簡単にモデルを試せる Gradio ベースの Web UI が含まれています。

1.  **Web UI の起動**
    環境構築とモデルのダウンロードが完了したら、以下のコマンドで Web UI を起動します。

    ```bash
    python app-web-ui.py
    ```

2.  **アクセス**
    ターミナルに表示される URL（通常は `http://127.0.0.1:7860`）をブラウザで開くと、インターフェースを使用できます。

#### コマンドラインでの直接実行

事前学習済みモデルをダウンロードし、環境をセットアップしたら、以下のコマンドでテストを直接実行することもできます。

```bash
python scripts/sr_val_ddpm_text_T_vqganfin_old.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt models/stablesr_turbo.ckpt --init-img input --outdir output --ddpm_steps 4 --dec_w 0.5 --seed 42 --n_samples 1 --vqgan_ckpt models/vqgan_cfw_00011.ckpt --colorfix_type wavelet
```

実行後、`output`ディレクトリに生成された画像が保存されます。

### Citation (引用)

If our work is useful for your research, please consider citing:

    @article{wang2024exploiting,
      author = {Wang, Jianyi and Yue, Zongsheng and Zhou, Shangchen and Chan, Kelvin C.K. and Loy, Chen Change},
      title = {Exploiting Diffusion Prior for Real-World Image Super-Resolution},
      article = {International Journal of Computer Vision},
      year = {2024}
    }

### License (ライセンス)

This project is licensed under <a rel="license" href="https://github.com/IceClear/StableSR/blob/main/LICENSE.txt">NTU S-Lab License 1.0</a>. Redistribution and use should follow this license.
