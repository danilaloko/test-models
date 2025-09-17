import os
from dotenv import load_dotenv
from diffusers import AutoPipelineForText2Image
import torch

# Загружаем переменные окружения из .env файла
load_dotenv()

# Загружаем модель с аутентификацией
pipeline = AutoPipelineForText2Image.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.bfloat16,
    use_auth_token=os.environ["HF_TOKEN"]  # Добавляем токен для доступа
)

# Перемещаем на GPU если доступно
if torch.cuda.is_available():
    pipeline = pipeline.to("cuda")
else:
    print("⚠️  CUDA недоступна, используем CPU (будет медленно)")

pipeline.load_lora_weights("lustlyai/Flux_Lustly.ai_Uncensored_nsfw_v1",
                           weight_name="flux_lustly-ai_v1.safetensors",
                           adapter_name="v1")

pipeline.set_adapters(["v1"], adapter_weights=[1])

prompt = "high quality, on the street, a naked woman with purple hair, standing next to a black naked male with big flaccid dick, both hold a neon sign 'lustly.ai'"

out = pipeline(
    prompt=prompt,
    guidance_scale=4,
    height=768,
    width=768,
    num_inference_steps=20,
).images[0]
display(out)
out.save("generated_images/lux_generated_image.png")
print("Lux изображение сохранено как 'generated_images/lux_generated_image.png'")