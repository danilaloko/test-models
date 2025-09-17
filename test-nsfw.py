import os
from dotenv import load_dotenv
from diffusers import DiffusionPipeline
import torch

# Загружаем переменные окружения из .env файла
load_dotenv()

# Загружаем модель
pipe = DiffusionPipeline.from_pretrained(
    "UnfilteredAI/NSFW-gen-v2",
    torch_dtype=torch.float16,
    use_auth_token=os.environ["HF_TOKEN"]
)

# Перемещаем на GPU если доступно
if torch.cuda.is_available():
    pipe = pipe.to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt).images[0]

# Сохраняем изображение
image.save("generated_images/nsfw_generated_image.png")
print("NSFW изображение сохранено как 'generated_images/nsfw_generated_image.png'")