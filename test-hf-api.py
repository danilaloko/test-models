import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Загружаем переменные окружения из .env файла
load_dotenv()

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
)

# Используем модель через Hugging Face Inference API
# Это намного быстрее и не требует загрузки модели локально
image = client.text_to_image(
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    model="stabilityai/stable-diffusion-xl-base-1.0",  # Модель, которая точно работает через API
    # model="runwayml/stable-diffusion-v1-5",  # Альтернатива
    # model="CompVis/stable-diffusion-v1-4",   # Еще одна альтернатива
)

# Сохраняем изображение
image.save("generated_images/hf_api_generated_image.png")
print("Изображение сгенерировано через Hugging Face API и сохранено как 'generated_images/hf_api_generated_image.png'")
