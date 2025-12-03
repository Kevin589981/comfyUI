import time
import torch
from diffusers import ZImagePipeline
from diffusers.models import AutoencoderKL, ZImageTransformer2DModel
from transformers import T5EncoderModel

def main():
    # 1. 设置设备和模型仓库ID
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    original_repo = "Tongyi-MAI/Z-Image-Turbo"
    quantized_repo = "drbaph/Z-Image-Turbo-FP8"
    
    # 2. 手动加载各个模型组件
    # 注意：我们不再在 ZImagePipeline.from_pretrained 中指定 dtype，
    # 因为FP8模型加载时不需要，而其他组件会根据其自身配置加载。

    print("Loading VAE from original repo...")
    vae = AutoencoderKL.from_pretrained(original_repo, subfolder="vae")

    print("Loading Text Encoder from original repo...")
    t5 = T5EncoderModel.from_pretrained(original_repo, subfolder="text_encoders")

    print(f"Loading Quantized FP8 Transformer from {quantized_repo}...")
    # 这里我们加载e4m3fn变体，它在质量和性能之间取得了很好的平衡
    transformer = ZImageTransformer2DModel.from_pretrained(quantized_repo, subfolder="z_image_turbo_fp8_e4m3fn.safetensors")

    # 3. 手动组装 Pipeline
    print("Assembling the pipeline...")
    pipe = ZImagePipeline(
        transformer=transformer,
        t5=t5,
        vae=vae,
    )
    
    # 4. 启用模型CPU卸载（仍然是保险措施）
    print("Enabling model CPU offload to save memory...")
    pipe.enable_model_cpu_offload()

    # 5. 定义提示词和生成图片
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

    # 6. 保存图片
    image = result.images[0]
    output_path = "generated_image.png"
    image.save(output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    main()