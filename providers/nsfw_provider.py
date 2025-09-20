"""
Провайдер для генерации изображений с помощью локальных NSFW моделей
"""

import os
import torch
import asyncio
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from diffusers import DiffusionPipeline

from .base_provider import BaseProvider

class NSFWProvider(BaseProvider):
    """Провайдер для работы с локальными NSFW моделями"""
    
    def __init__(self):
        super().__init__(
            name="NSFW Local",
            description="Локальная генерация с помощью NSFW моделей (требует GPU)"
        )
        self.pipeline: Optional[DiffusionPipeline] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.available_models = [
            "UnfilteredAI/NSFW-gen-v2",
            "UnfilteredAI/NSFW-3B"
        ]
        self.current_model = self.available_models[0]
    
    async def initialize(self) -> bool:
        """Инициализация провайдера"""
        try:
            load_dotenv()
            hf_token = os.getenv("HF_TOKEN")
            
            if not hf_token:
                print("❌ HF_TOKEN не найден в переменных окружения")
                return False
            
            if not torch.cuda.is_available():
                print("⚠️ CUDA недоступна, будет использоваться CPU (медленно)")
                self.device = "cpu"
            
            print(f"🔄 Загружаю модель {self.current_model}...")
            
            # Загружаем модель в отдельном потоке
            self.pipeline = await self._run_sync_in_async(
                self._load_pipeline,
                model_id=self.current_model,
                hf_token=hf_token
            )
            
            if self.pipeline is None:
                return False
            
            self.is_initialized = True
            print(f"✅ {self.name} инициализирован успешно на {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка инициализации {self.name}: {e}")
            return False
    
    def _load_pipeline(self, model_id: str, hf_token: str) -> Optional[DiffusionPipeline]:
        """Синхронная загрузка пайплайна"""
        try:
            # Пробуем разные конфигурации загрузки
            load_configs = [
                {
                    "torch_dtype": torch.float16,
                    "use_auth_token": hf_token,
                    "low_cpu_mem_usage": True
                },
                {
                    "torch_dtype": torch.bfloat16,
                    "use_auth_token": hf_token,
                    "low_cpu_mem_usage": True
                },
                {
                    "torch_dtype": torch.float32,
                    "use_auth_token": hf_token,
                    "low_cpu_mem_usage": True
                }
            ]
            
            for i, config in enumerate(load_configs):
                try:
                    print(f"Попытка загрузки {i+1}/{len(load_configs)}...")
                    pipeline = DiffusionPipeline.from_pretrained(model_id, **config)
                    
                    # Перемещаем на устройство
                    if self.device == "cuda":
                        pipeline = pipeline.to("cuda")
                    
                    print(f"✅ Модель загружена успешно (конфигурация {i+1})")
                    return pipeline
                    
                except Exception as e:
                    print(f"❌ Конфигурация {i+1} не сработала: {e}")
                    continue
            
            print("❌ Все попытки загрузки провалились")
            return None
            
        except Exception as e:
            print(f"❌ Критическая ошибка загрузки: {e}")
            return None
    
    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Генерация изображения с помощью NSFW модели
        
        Args:
            prompt: Текстовое описание изображения
            **kwargs: Дополнительные параметры:
                - width: int - ширина изображения (по умолчанию 512)
                - height: int - высота изображения (по умолчанию 512)
                - num_inference_steps: int - количество шагов (по умолчанию 20)
                - guidance_scale: float - сила следования промпту (по умолчанию 7.5)
                - num_images: int - количество изображений (по умолчанию 1)
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Провайдер не инициализирован или недоступен"
            }
        
        try:
            # Извлекаем параметры
            width = kwargs.get("width", 512)
            height = kwargs.get("height", 512)
            num_inference_steps = kwargs.get("num_inference_steps", 20)
            guidance_scale = kwargs.get("guidance_scale", 7.5)
            num_images = kwargs.get("num_images", 1)
            
            # Генерируем имя файла
            image_path = self._generate_filename(prompt)
            
            print(f"🎨 Генерирую изображение через {self.current_model}...")
            print(f"📝 Промпт: {prompt}")
            print(f"🖥️ Устройство: {self.device}")
            
            # Генерируем изображение в отдельном потоке
            result = await self._run_sync_in_async(
                self._generate_image_sync,
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images=num_images
            )
            
            if result is None:
                return {
                    "success": False,
                    "error": "Ошибка генерации изображения"
                }
            
            # Сохраняем первое изображение
            result.images[0].save(image_path)
            
            print(f"✅ Изображение сохранено: {image_path}")
            
            return {
                "success": True,
                "image_path": image_path,
                "metadata": {
                    "model": self.current_model,
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "num_images": num_images,
                    "device": self.device,
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
    
    def _generate_image_sync(self, prompt: str, width: int, height: int, 
                           num_inference_steps: int, guidance_scale: float, num_images: int):
        """Синхронная генерация изображения"""
        try:
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images
                )
            return result
        except Exception as e:
            print(f"❌ Ошибка синхронной генерации: {e}")
            return None
    
    def is_available(self) -> bool:
        """Проверяет доступность провайдера"""
        return self.is_initialized and self.pipeline is not None
    
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
            # Перезагружаем пайплайн с новой моделью
            self.is_initialized = False
            self.pipeline = None
            return True
        return False
    
    def get_available_models(self) -> list:
        """Возвращает список доступных моделей"""
        return self.available_models.copy()
    
    def get_current_model(self) -> str:
        """Возвращает текущую модель"""
        return self.current_model
    
    def get_device(self) -> str:
        """Возвращает используемое устройство"""
        return self.device
