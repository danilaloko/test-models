"""
Менеджер для управления провайдерами моделей генерации изображений
"""

import asyncio
from typing import Dict, Optional, List
from .base_provider import BaseProvider

class ProviderManager:
    """Менеджер для управления всеми провайдерами"""
    
    def __init__(self):
        self.providers: Dict[str, BaseProvider] = {}
        self.initialization_status: Dict[str, bool] = {}
        self.initialization_in_progress: Dict[str, bool] = {}
    
    def register_provider(self, provider_id: str, provider: BaseProvider) -> bool:
        """
        Регистрирует провайдер
        
        Args:
            provider_id: Уникальный идентификатор провайдера
            provider: Экземпляр провайдера
            
        Returns:
            True если провайдер зарегистрирован успешно
        """
        try:
            if provider_id in self.providers:
                print(f"⚠️ Провайдер {provider_id} уже зарегистрирован, заменяем...")
            
            self.providers[provider_id] = provider
            self.initialization_status[provider_id] = False
            self.initialization_in_progress[provider_id] = False
            print(f"✅ Провайдер {provider_id} ({provider.get_name()}) зарегистрирован")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка регистрации провайдера {provider_id}: {e}")
            return False
    
    def unregister_provider(self, provider_id: str) -> bool:
        """
        Удаляет провайдер из менеджера
        
        Args:
            provider_id: Идентификатор провайдера
            
        Returns:
            True если провайдер удален успешно
        """
        if provider_id in self.providers:
            del self.providers[provider_id]
            if provider_id in self.initialization_status:
                del self.initialization_status[provider_id]
            if provider_id in self.initialization_in_progress:
                del self.initialization_in_progress[provider_id]
            print(f"✅ Провайдер {provider_id} удален")
            return True
        return False
    
    def get_provider(self, provider_id: str) -> Optional[BaseProvider]:
        """
        Возвращает провайдер по идентификатору
        
        Args:
            provider_id: Идентификатор провайдера
            
        Returns:
            Экземпляр провайдера или None
        """
        return self.providers.get(provider_id)
    
    def get_all_providers(self) -> Dict[str, BaseProvider]:
        """Возвращает все зарегистрированные провайдеры"""
        return self.providers.copy()
    
    def get_available_providers(self) -> Dict[str, BaseProvider]:
        """Возвращает только доступные провайдеры"""
        return {
            provider_id: provider 
            for provider_id, provider in self.providers.items()
            if provider.is_available()
        }
    
    async def initialize_provider_lazy(self, provider_id: str) -> bool:
        """
        Ленивая инициализация провайдера (только при необходимости)
        
        Args:
            provider_id: Идентификатор провайдера
            
        Returns:
            True если инициализация успешна
        """
        if provider_id not in self.providers:
            print(f"❌ Провайдер {provider_id} не найден")
            return False
        
        # Если уже инициализирован, возвращаем True
        if self.initialization_status.get(provider_id, False):
            return True
        
        # Если инициализация уже в процессе, ждем
        if self.initialization_in_progress.get(provider_id, False):
            print(f"⏳ Провайдер {provider_id} уже инициализируется...")
            # Ждем завершения инициализации
            while self.initialization_in_progress.get(provider_id, False):
                await asyncio.sleep(0.1)
            return self.initialization_status.get(provider_id, False)
        
        # Запускаем инициализацию
        return await self.initialize_provider(provider_id)
    
    async def initialize_provider(self, provider_id: str) -> bool:
        """
        Инициализирует конкретный провайдер
        
        Args:
            provider_id: Идентификатор провайдера
            
        Returns:
            True если инициализация успешна
        """
        if provider_id not in self.providers:
            print(f"❌ Провайдер {provider_id} не найден")
            return False
        
        provider = self.providers[provider_id]
        
        # Отмечаем, что инициализация началась
        self.initialization_in_progress[provider_id] = True
        
        try:
            print(f"🔄 Инициализация провайдера {provider_id}...")
            success = await provider.initialize()
            self.initialization_status[provider_id] = success
            
            if success:
                print(f"✅ Провайдер {provider_id} инициализирован успешно")
            else:
                print(f"❌ Ошибка инициализации провайдера {provider_id}")
            
            return success
            
        except Exception as e:
            print(f"❌ Критическая ошибка инициализации {provider_id}: {e}")
            self.initialization_status[provider_id] = False
            return False
        finally:
            # Отмечаем, что инициализация завершена
            self.initialization_in_progress[provider_id] = False
    
    async def initialize_all_providers(self) -> Dict[str, bool]:
        """
        Инициализирует все зарегистрированные провайдеры
        
        Returns:
            Словарь с результатами инициализации для каждого провайдера
        """
        print("🔄 Инициализация всех провайдеров...")
        
        # Создаем задачи для параллельной инициализации
        tasks = []
        provider_ids = list(self.providers.keys())
        
        for provider_id in provider_ids:
            task = asyncio.create_task(self.initialize_provider(provider_id))
            tasks.append((provider_id, task))
        
        # Ждем завершения всех задач
        results = {}
        for provider_id, task in tasks:
            try:
                result = await task
                results[provider_id] = result
            except Exception as e:
                print(f"❌ Ошибка инициализации {provider_id}: {e}")
                results[provider_id] = False
        
        # Выводим итоговую статистику
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        print(f"📊 Инициализация завершена: {successful}/{total} провайдеров успешно")
        
        return results
    
    def get_provider_status(self) -> Dict[str, Dict[str, any]]:
        """
        Возвращает статус всех провайдеров
        
        Returns:
            Словарь с информацией о каждом провайдере
        """
        status = {}
        
        for provider_id, provider in self.providers.items():
            is_initialized = self.initialization_status.get(provider_id, False)
            is_initializing = self.initialization_in_progress.get(provider_id, False)
            
            status[provider_id] = {
                "name": provider.get_name(),
                "description": provider.get_description(),
                "is_available": provider.is_available() if is_initialized else False,
                "is_initialized": is_initialized,
                "is_initializing": is_initializing,
                "status": self._get_provider_status_text(provider_id, is_initialized, is_initializing)
            }
        
        return status
    
    def _get_provider_status_text(self, provider_id: str, is_initialized: bool, is_initializing: bool) -> str:
        """
        Возвращает текстовое описание статуса провайдера
        
        Args:
            provider_id: Идентификатор провайдера
            is_initialized: Инициализирован ли провайдер
            is_initializing: Инициализируется ли сейчас
            
        Returns:
            Текстовое описание статуса
        """
        if is_initializing:
            return "🔄 Загружается..."
        elif is_initialized:
            provider = self.providers.get(provider_id)
            if provider and provider.is_available():
                return "✅ Доступен"
            else:
                return "❌ Недоступен"
        else:
            return "⏸️ Не загружен"
    
    def get_provider_info(self, provider_id: str) -> Optional[Dict[str, any]]:
        """
        Возвращает информацию о конкретном провайдере
        
        Args:
            provider_id: Идентификатор провайдера
            
        Returns:
            Словарь с информацией о провайдере или None
        """
        if provider_id not in self.providers:
            return None
        
        provider = self.providers[provider_id]
        
        info = {
            "id": provider_id,
            "name": provider.get_name(),
            "description": provider.get_description(),
            "is_available": provider.is_available(),
            "is_initialized": self.initialization_status.get(provider_id, False)
        }
        
        # Добавляем специфичную информацию для разных типов провайдеров
        if hasattr(provider, 'get_available_models'):
            info["available_models"] = provider.get_available_models()
        
        if hasattr(provider, 'get_current_model'):
            info["current_model"] = provider.get_current_model()
        
        if hasattr(provider, 'get_device'):
            info["device"] = provider.get_device()
        
        if hasattr(provider, 'get_available_loras'):
            info["available_loras"] = provider.get_available_loras()
        
        if hasattr(provider, 'get_current_lora'):
            info["current_lora"] = provider.get_current_lora()
        
        return info
    
    async def generate_image(self, provider_id: str, prompt: str, **kwargs) -> Dict[str, any]:
        """
        Генерирует изображение с помощью указанного провайдера
        
        Args:
            provider_id: Идентификатор провайдера
            prompt: Текстовое описание изображения
            **kwargs: Дополнительные параметры для генерации
            
        Returns:
            Результат генерации
        """
        provider = self.get_provider(provider_id)
        
        if not provider:
            return {
                "success": False,
                "error": f"Провайдер {provider_id} не найден"
            }
        
        # Инициализируем провайдер если нужно
        if not self.initialization_status.get(provider_id, False):
            success = await self.initialize_provider_lazy(provider_id)
            if not success:
                return {
                    "success": False,
                    "error": f"Не удалось инициализировать провайдер {provider_id}"
                }
        
        if not provider.is_available():
            return {
                "success": False,
                "error": f"Провайдер {provider_id} недоступен"
            }
        
        return await provider.generate_image(prompt, **kwargs)
    
    def list_providers(self) -> List[str]:
        """Возвращает список всех зарегистрированных провайдеров"""
        return list(self.providers.keys())
    
    def get_provider_count(self) -> int:
        """Возвращает количество зарегистрированных провайдеров"""
        return len(self.providers)
    
    def get_available_provider_count(self) -> int:
        """Возвращает количество доступных провайдеров"""
        return len(self.get_available_providers())
