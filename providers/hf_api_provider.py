"""
Провайдер для генерации изображений через Hugging Face API
"""

import os
import asyncio
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from .base_provider import BaseProvider

class HFApiProvider(BaseProvider):
    """Провайдер для работы с Hugging Face Inference API"""
    
    def __init__(self):
        super().__init__(
            name="Hugging Face API",
            description="Быстрая генерация через Hugging Face Inference API (Stable Diffusion XL)"
        )
        self.client: Optional[InferenceClient] = None
        self.available_models = [
            "stabilityai/stable-diffusion-xl-base-1.0",
            "runwayml/stable-diffusion-v1-5",
            "CompVis/stable-diffusion-v1-4"
        ]
        self.current_model = self.available_models[0]  # По умолчанию SDXL
    
    async def initialize(self) -> bool:
        """Инициализация провайдера"""
        try:
            load_dotenv()
            hf_token = os.getenv("HF_TOKEN")
            
            if not hf_token:
                print("❌ HF_TOKEN не найден в переменных окружения")
                return False
            
            # Создаем клиент в отдельном потоке
            self.client = await self._run_sync_in_async(
                lambda: InferenceClient(api_key=hf_token)
            )
            
            self.is_initialized = True
            print(f"✅ {self.name} инициализирован успешно")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка инициализации {self.name}: {e}")
            return False
    
    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Генерация изображения через Hugging Face API
        
        Args:
            prompt: Текстовое описание изображения
            **kwargs: Дополнительные параметры:
                - model: str - модель для использования
                - width: int - ширина изображения (по умолчанию 1024)
                - height: int - высота изображения (по умолчанию 1024)
                - num_inference_steps: int - количество шагов (по умолчанию 20)
                - guidance_scale: float - сила следования промпту (по умолчанию 7.5)
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Провайдер не инициализирован или недоступен"
            }
        
        try:
            # Извлекаем параметры
            model = kwargs.get("model", self.current_model)
            width = kwargs.get("width", 1024)
            height = kwargs.get("height", 1024)
            num_inference_steps = kwargs.get("num_inference_steps", 20)
            guidance_scale = kwargs.get("guidance_scale", 7.5)
            
            # Генерируем имя файла
            image_path = self._generate_filename(prompt)
            
            print(f"🎨 Генерирую изображение через {model}...")
            print(f"📝 Промпт: {prompt}")
            
            # Генерируем изображение в отдельном потоке
            image = await self._run_sync_in_async(
                self._generate_image_sync,
                prompt=prompt,
                model=model,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            
            # Сохраняем изображение
            await self._run_sync_in_async(image.save, image_path)
            
            print(f"✅ Изображение сохранено: {image_path}")
            
            return {
                "success": True,
                "image_path": image_path,
                "metadata": {
                    "model": model,
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "provider": self.name
                }
            }
            
        except Exception as e:
            error_msg = f"Ошибка генерации через {self.name}: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
    
    def _generate_image_sync(self, prompt: str, model: str, width: int, height: int, 
                           num_inference_steps: int, guidance_scale: float):
        """Синхронная генерация изображения"""
        return self.client.text_to_image(
            prompt=prompt,
            model=model,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
    
    def is_available(self) -> bool:
        """Проверяет доступность провайдера"""
        return self.is_initialized and self.client is not None
    
    def set_model(self, model: str) -> bool:
        """
        Устанавливает модель для генерации
        
        Args:
            model: Название модели
            
        Returns:
            True если модель установлена успешно
        """
        if model in self.available_models:
            self.current_model = model
            return True
        return False
    
    def get_available_models(self) -> list:
        """Возвращает список доступных моделей"""
        return self.available_models.copy()
    
    def get_current_model(self) -> str:
        """Возвращает текущую модель"""
        return self.current_model
