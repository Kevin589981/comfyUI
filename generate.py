import time
import torch
from diffusers import ZImagePipeline

def main():
    # 1. 设置设备
    # 在GitHub Actions的macOS runner上，我们使用 'mps'
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. 加载模型
    # 使用 bfloat16 以获得最佳性能
    # low_cpu_mem_usage=False 在有足够RAM时可以加快加载速度
    print("Loading Z-Image-Turbo pipeline...")
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to(device)
    print("Pipeline loaded successfully.")

    # 3. 定义提示词
    prompt = "masterpiece, best quality, a futuristic cybernetic city skyline at dusk, neon lights reflecting on wet streets, flying vehicles weaving through holographic advertisements"
    
    # 4. 生成图片
    print(f"Generating image for prompt: '{prompt}'")
    start_time = time.time()
    
    # 注意：在MPS上，需要为生成器明确指定CPU设备
    generator = torch.Generator(device="cpu").manual_seed(42)

    result = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=4,  # Z-Image Turbo 只需要很少的步数
        guidance_scale=0.0,
        generator=generator,
    )

    end_time = time.time()
    print(f"Inference took: {end_time - start_time:.2f} seconds")

    # 5. 保存图片
    image = result.images[0]
    output_path = "generated_image.png"
    image.save(output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    main()