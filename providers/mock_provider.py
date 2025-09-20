"""
Провайдер-заглушка для тестирования логики бота без реальных моделей
"""

import os
import asyncio
import random
from typing import Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
import textwrap

from .base_provider import BaseProvider

class MockProvider(BaseProvider):
    """Провайдер-заглушка для тестирования"""
    
    def __init__(self):
        super().__init__(
            name="Mock Provider",
            description="Тестовый провайдер для проверки логики бота (генерирует простые изображения)"
        )
        self.available_models = [
            "mock-stable-diffusion-v1",
            "mock-flux-dev",
            "mock-nsfw-v2",
            "mock-dalle-3"
        ]
        self.current_model = self.available_models[0]
        self.generation_delay = 2  # Задержка в секундах для имитации генерации
    
    async def initialize(self) -> bool:
        """Инициализация провайдера-заглушки"""
        try:
            # Имитируем задержку инициализации
            await asyncio.sleep(0.5)
            
            self.is_initialized = True
            print(f"✅ {self.name} инициализирован успешно (заглушка)")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка инициализации {self.name}: {e}")
            return False
    
    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Генерирует тестовое изображение
        
        Args:
            prompt: Текстовое описание изображения
            **kwargs: Дополнительные параметры:
                - width: int - ширина изображения (по умолчанию 512)
                - height: int - высота изображения (по умолчанию 512)
                - model: str - модель для использования
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
            model = kwargs.get("model", self.current_model)
            
            # Ограничиваем размеры для тестирования
            width = min(max(width, 256), 1024)
            height = min(max(height, 256), 1024)
            
            # Генерируем имя файла
            image_path = self._generate_filename(prompt)
            
            print(f"🎨 [MOCK] Генерирую изображение через {model}...")
            print(f"📝 [MOCK] Промпт: {prompt}")
            print(f"⏳ [MOCK] Имитирую генерацию ({self.generation_delay}с)...")
            
            # Имитируем задержку генерации
            await asyncio.sleep(self.generation_delay)
            
            # Генерируем тестовое изображение в отдельном потоке
            await self._run_sync_in_async(
                self._generate_mock_image,
                prompt=prompt,
                width=width,
                height=height,
                model=model,
                image_path=image_path
            )
            
            print(f"✅ [MOCK] Изображение сохранено: {image_path}")
            
            return {
                "success": True,
                "image_path": image_path,
                "metadata": {
                    "model": model,
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "provider": self.name,
                    "is_mock": True,
                    "generation_time": self.generation_delay
                }
            }
            
        except Exception as e:
            error_msg = f"Ошибка генерации через {self.name}: {str(e)}"
            print(f"❌ [MOCK] {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
    
    def _generate_mock_image(self, prompt: str, width: int, height: int, model: str, image_path: str):
        """Синхронная генерация тестового изображения"""
        try:
            # Создаем изображение с градиентным фоном
            image = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(image)
            
            # Создаем градиентный фон
            for y in range(height):
                # Создаем градиент от синего к фиолетовому
                r = int(50 + (y / height) * 100)
                g = int(50 + (y / height) * 50)
                b = int(150 + (y / height) * 100)
                draw.line([(0, y), (width, y)], fill=(r, g, b))
            
            # Добавляем рамку
            draw.rectangle([10, 10, width-10, height-10], outline=(255, 255, 255), width=3)
            
            # Добавляем текст с информацией
            try:
                # Пытаемся использовать системный шрифт
                font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except:
                # Если системный шрифт недоступен, используем стандартный
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
            
            # Заголовок
            title = f"Mock Image - {model}"
            title_bbox = draw.textbbox((0, 0), title, font=font_large)
            title_width = title_bbox[2] - title_bbox[0]
            title_x = (width - title_width) // 2
            draw.text((title_x, 30), title, fill=(255, 255, 255), font=font_large)
            
            # Промпт (разбиваем на строки если длинный)
            wrapped_prompt = textwrap.fill(prompt, width=40)
            prompt_lines = wrapped_prompt.split('\n')
            
            # Позиция для текста промпта
            text_y = 80
            for line in prompt_lines:
                line_bbox = draw.textbbox((0, 0), line, font=font_small)
                line_width = line_bbox[2] - line_bbox[0]
                line_x = (width - line_width) // 2
                draw.text((line_x, text_y), line, fill=(255, 255, 255), font=font_small)
                text_y += 25
            
            # Добавляем декоративные элементы
            # Круги в углах
            circle_radius = 20
            draw.ellipse([20, 20, 20 + circle_radius*2, 20 + circle_radius*2], 
                        outline=(255, 255, 255), width=2)
            draw.ellipse([width-40, 20, width-40 + circle_radius*2, 20 + circle_radius*2], 
                        outline=(255, 255, 255), width=2)
            draw.ellipse([20, height-40, 20 + circle_radius*2, height-40 + circle_radius*2], 
                        outline=(255, 255, 255), width=2)
            draw.ellipse([width-40, height-40, width-40 + circle_radius*2, height-40 + circle_radius*2], 
                        outline=(255, 255, 255), width=2)
            
            # Добавляем случайные точки для "шума"
            for _ in range(50):
                x = random.randint(0, width)
                y = random.randint(0, height)
                color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
                draw.point((x, y), fill=color)
            
            # Информация о размере
            size_info = f"{width}x{height}"
            size_bbox = draw.textbbox((0, 0), size_info, font=font_small)
            size_width = size_bbox[2] - size_bbox[0]
            size_x = (width - size_width) // 2
            draw.text((size_x, height - 40), size_info, fill=(255, 255, 255), font=font_small)
            
            # Сохраняем изображение
            image.save(image_path, 'PNG')
            
        except Exception as e:
            print(f"❌ [MOCK] Ошибка создания изображения: {e}")
            # Создаем простое изображение в случае ошибки
            simple_image = Image.new('RGB', (width, height), color=(100, 150, 200))
            simple_image.save(image_path, 'PNG')
    
    def is_available(self) -> bool:
        """Проверяет доступность провайдера"""
        return self.is_initialized
    
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
    
    def set_generation_delay(self, delay: float):
        """
        Устанавливает задержку генерации для тестирования
        
        Args:
            delay: Задержка в секундах
        """
        self.generation_delay = max(0.1, delay)
    
    def get_generation_delay(self) -> float:
        """Возвращает текущую задержку генерации"""
        return self.generation_delay
