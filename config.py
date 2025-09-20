"""
Конфигурационный файл для Telegram бота генерации изображений
"""

import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

class Config:
    """Класс конфигурации приложения"""
    
    # Telegram Bot
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    
    # Hugging Face
    HF_TOKEN = os.getenv('HF_TOKEN')
    
    # Настройки генерации изображений
    DEFAULT_IMAGE_WIDTH = 1024
    DEFAULT_IMAGE_HEIGHT = 1024
    DEFAULT_NUM_INFERENCE_STEPS = 20
    DEFAULT_GUIDANCE_SCALE = 7.5
    
    # Настройки для разных провайдеров
    HF_API_SETTINGS = {
        'default_model': 'stabilityai/stable-diffusion-xl-base-1.0',
        'width': 1024,
        'height': 1024,
        'num_inference_steps': 20,
        'guidance_scale': 7.5
    }
    
    NSFW_SETTINGS = {
        'default_model': 'UnfilteredAI/NSFW-gen-v2',
        'width': 512,
        'height': 512,
        'num_inference_steps': 20,
        'guidance_scale': 7.5
    }
    
    FLUX_SETTINGS = {
        'default_model': 'black-forest-labs/FLUX.1-dev',
        'width': 768,
        'height': 768,
        'num_inference_steps': 20,
        'guidance_scale': 4.0,
        'default_lora': 'lustlyai/Flux_Lustly.ai_Uncensored_nsfw_v1'
    }
    
    # Пути
    GENERATED_IMAGES_DIR = 'generated_images'
    MODELS_DIR = 'models'
    
    # Лимиты
    MAX_PROMPT_LENGTH = 1000
    MAX_IMAGE_SIZE = 2048
    MIN_IMAGE_SIZE = 256
    
    # Таймауты (в секундах)
    GENERATION_TIMEOUT = 300  # 5 минут
    INITIALIZATION_TIMEOUT = 600  # 10 минут
    
    @classmethod
    def validate(cls) -> bool:
        """
        Проверяет корректность конфигурации
        
        Returns:
            True если конфигурация корректна
        """
        errors = []
        
        if not cls.TELEGRAM_BOT_TOKEN:
            errors.append("TELEGRAM_BOT_TOKEN не найден в переменных окружения")
        
        if not cls.HF_TOKEN:
            errors.append("HF_TOKEN не найден в переменных окружения")
        
        if errors:
            print("❌ Ошибки конфигурации:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    @classmethod
    def create_directories(cls):
        """Создает необходимые директории"""
        os.makedirs(cls.GENERATED_IMAGES_DIR, exist_ok=True)
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
    
    @classmethod
    def get_provider_settings(cls, provider_name: str) -> dict:
        """
        Возвращает настройки для конкретного провайдера
        
        Args:
            provider_name: Название провайдера
            
        Returns:
            Словарь с настройками провайдера
        """
        settings_map = {
            'hf_api': cls.HF_API_SETTINGS,
            'nsfw': cls.NSFW_SETTINGS,
            'flux': cls.FLUX_SETTINGS
        }
        
        return settings_map.get(provider_name, {})
