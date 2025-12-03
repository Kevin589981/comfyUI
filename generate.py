import time
import torch
from diffusers import ZImagePipeline

def main():
    # 1. 设置设备
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. 加载模型
    print("Loading Z-Image-Turbo pipeline...")
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        # 使用新的 'dtype' 参数，替换已弃用的 'torch_dtype'
        dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    print("Pipeline loaded from pretrained.")

    # --- 这是关键的解决方案 ---
    # 启用模型CPU卸载以在内存受限的环境中运行
    print("Enabling model CPU offload to save memory...")
    pipe.enable_model_cpu_offload()
    # -------------------------
    # 注意：在启用卸载后，不需要再手动调用 pipe.to(device)
    
    # 3. 定义提示词
    prompt = "masterpiece, best quality, a futuristic cybernetic city skyline at dusk, neon lights reflecting on wet streets, flying vehicles weaving through holographic advertisements"
    
    # 4. 生成图片
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

    # 5. 保存图片
    image = result.images[0]
    output_path = "generated_image.png"
    image.save(output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    main()