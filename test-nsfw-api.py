import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Загружаем переменные окружения из .env файла
load_dotenv()

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
)

# Попробуем использовать NSFW модель через API
# Внимание: не все NSFW модели доступны через Inference API
try:
    image = client.text_to_image(
        "beautiful woman, portrait, artistic, detailed",
        model="UnfilteredAI/NSFW-gen-v2",  # Попробуем через API
    )
    
    # Сохраняем изображение
    image.save("generated_images/nsfw_api_generated_image.png")
    print("NSFW изображение сгенерировано через API и сохранено как 'generated_images/nsfw_api_generated_image.png'")
    
except Exception as e:
    print(f"Ошибка при использовании NSFW модели через API: {e}")
    print("NSFW модели могут быть недоступны через Inference API.")
    print("Попробуйте использовать обычную модель:")
    
    # Fallback на обычную модель
    image = client.text_to_image(
        "beautiful woman, portrait, artistic, detailed",
        model="stabilityai/stable-diffusion-2-1",
    )
    
    image.save("generated_images/fallback_generated_image.png")
    print("Изображение сгенерировано обычной моделью и сохранено как 'generated_images/fallback_generated_image.png'")
