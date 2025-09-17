import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Загружаем переменные окружения из .env файла
load_dotenv()

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
)

# output is a PIL.Image object
# Используем модель, которая точно работает через Inference API
image = client.text_to_image(
    "Astronaut riding a horse",
    model="stabilityai/stable-diffusion-xl-base-1.0",  # Модель, которая работает через API
    # model="runwayml/stable-diffusion-v1-5",  # Альтернатива
    # model="CompVis/stable-diffusion-v1-4",   # Еще одна альтернатива
    # model="Qwen/Qwen-Image",  # Оригинальная модель
)

# Сохраняем изображение
image.save("generated_images/generated_image.png")
print("Изображение сохранено как 'generated_images/generated_image.png'")