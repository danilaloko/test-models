import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Загружаем переменные окружения из .env файла
load_dotenv()

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
)

# output is a PIL.Image object
# Попробуйте одну из этих моделей:
image = client.text_to_image(
    "Astronaut riding a horse",
    model="stabilityai/stable-diffusion-2-1",  # Популярная модель
    # model="runwayml/stable-diffusion-v1-5",  # Альтернатива
    # model="CompVis/stable-diffusion-v1-4",   # Еще одна альтернатива
    # model="Qwen/Qwen-Image",  # Оригинальная модель
)

# Сохраняем изображение
image.save("generated_images/generated_image.png")
print("Изображение сохранено как 'generated_images/generated_image.png'")