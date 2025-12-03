import time
import torch
from diffusers import ZImagePipeline
from diffusers.models import AutoencoderKL, ZImageTransformer2DModel
from transformers import T5EncoderModel
from huggingface_hub import snapshot_download

def main():
    # 1. 设置设备和模型仓库ID
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    original_repo = "Tongyi-MAI/Z-Image-Turbo"
    quantized_repo = "drbaph/Z-Image-Turbo-FP8"
    
    # 2. 先使用 snapshot_download 确保所有组件都已下载到本地缓存
    #    这可以避免因远程仓库文件命名不规范导致的加载问题
    print("Downloading model components from Hugging Face Hub...")
    
    # snapshot_download 会返回一个指向包含所有文件的本地目录的路径
    local_model_path = snapshot_download(
        repo_id=original_repo,
        allow_patterns=["vae/*", "text_encoders/*"]
    )
    
    local_quant_path = snapshot_download(
        repo_id=quantized_repo,
        allow_patterns=["z_image_turbo_fp8_e4m3fn.safetensors/*"]
    )

    print("Model components downloaded. Now loading from local cache...")

    # 3. 从本地缓存路径加载各个模型组件
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(local_model_path, subfolder="vae")

    print("Loading Text Encoder...")
    # 现在我们从确定的本地路径加载，OSError将不复存在
    t5 = T5EncoderModel.from_pretrained(local_model_path, subfolder="text_encoders")

    print("Loading Quantized FP8 Transformer...")
    transformer = ZImageTransformer2DModel.from_pretrained(local_quant_path, subfolder="z_image_turbo_fp8_e4m3fn.safetensors")

    # 4. 手动组装 Pipeline
    print("Assembling the pipeline...")
    pipe = ZImagePipeline(
        transformer=transformer,
        t5=t5,
        vae=vae,
    )
    
    # 5. 启用模型CPU卸载
    print("Enabling model CPU offload to save memory...")
    pipe.enable_model_cpu_offload()

    # 6. 定义提示词和生成图片
    prompt = "masterpiece, best quality, a futuristic cybernetic city skyline at dusk, neon lights reflecting on wet streets, flying vehicles weaving through holographic advertisements"
    
    print(f"Generating image for prompt: '{prompt}'")
    start_time = time.time()
    
    generator = torch.Generator(device="cpu").manual_seed(42)

    result = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=4,
        guidance_scale=0.0,
        generator=generator,
    )

    end_time = time.time()
    print(f"Inference took: {end_time - start_time:.2f} seconds")

    # 7. 保存图片
    image = result.images[0]
    output_path = "generated_image.png"
    image.save(output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    main()