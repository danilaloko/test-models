"""
Базовый абстрактный класс для провайдеров моделей генерации изображений
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio
import os
from datetime import datetime

class BaseProvider(ABC):
    """Базовый класс для всех провайдеров моделей"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Инициализация провайдера
        Возвращает True если инициализация успешна
        """
        pass
    
    @abstractmethod
    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Генерация изображения по промпту
        
        Args:
            prompt: Текстовое описание изображения
            **kwargs: Дополнительные параметры для конкретной модели
            
        Returns:
            Dict с результатами:
            - success: bool - успешность генерации
            - image_path: str - путь к сгенерированному изображению (если success=True)
            - error: str - сообщение об ошибке (если success=False)
            - metadata: dict - дополнительная информация о генерации
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Проверяет доступность провайдера
        Возвращает True если провайдер готов к работе
        """
        pass
    
    def get_name(self) -> str:
        """Возвращает название провайдера"""
        return self.name
    
    def get_description(self) -> str:
        """Возвращает описание провайдера"""
        return self.description
    
    def _generate_filename(self, prompt: str, extension: str = "png") -> str:
        """
        Генерирует уникальное имя файла для изображения
        
        Args:
            prompt: Промпт для генерации
            extension: Расширение файла
            
        Returns:
            Путь к файлу
        """
        # Создаем директорию если не существует
        os.makedirs("generated_images", exist_ok=True)
        
        # Генерируем имя файла на основе времени и части промпта
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c for c in prompt[:20] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_prompt = safe_prompt.replace(' ', '_')
        
        filename = f"{self.name}_{timestamp}_{safe_prompt}.{extension}"
        return os.path.join("generated_images", filename)
    
    def _run_sync_in_async(self, func, *args, **kwargs):
        """
        Запускает синхронную функцию в асинхронном контексте
        
        Args:
            func: Синхронная функция для выполнения
            *args: Аргументы функции
            **kwargs: Ключевые аргументы функции
            
        Returns:
            Результат выполнения функции
        """
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(None, lambda: func(*args, **kwargs))
