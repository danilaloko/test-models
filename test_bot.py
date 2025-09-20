#!/usr/bin/env python3
"""
Скрипт для тестирования Telegram бота без реального Telegram API
"""

import asyncio
import os
from dotenv import load_dotenv
from providers.provider_manager import ProviderManager
from providers.mock_provider import MockProvider
from providers.hf_api_provider import HFApiProvider
from providers.nsfw_provider import NSFWProvider
from providers.flux_provider import FluxProvider

async def test_providers():
    """Тестирует все провайдеры"""
    print("🧪 Тестирование провайдеров...")
    
    # Создаем менеджер провайдеров
    manager = ProviderManager()
    
    # Регистрируем провайдеры
    manager.register_provider('mock', MockProvider())
    manager.register_provider('hf_api', HFApiProvider())
    manager.register_provider('nsfw', NSFWProvider())
    manager.register_provider('flux', FluxProvider())
    
    # Инициализируем только mock провайдер
    print("\n🔄 Инициализация mock провайдера...")
    mock_success = await manager.initialize_provider('mock')
    print(f"Mock провайдер: {'✅ Инициализирован' if mock_success else '❌ Не инициализирован'}")
    
    # Показываем статус всех провайдеров
    print("\n📊 Статус провайдеров:")
    provider_status = manager.get_provider_status()
    for provider_id, status_info in provider_status.items():
        provider = manager.get_provider(provider_id)
        print(f"{status_info['status']} {provider.get_name()}")
    
    # Тестируем доступные провайдеры
    available_providers = manager.get_available_providers()
    print(f"\n🎯 Доступно провайдеров: {len(available_providers)}")
    
    # Тестируем mock провайдер (должен быть доступен)
    if 'mock' in available_providers:
        print(f"\n🧪 Тестирование Mock провайдера...")
        test_prompt = "beautiful landscape with mountains and lake"
        
        try:
            result = await manager.generate_image('mock', test_prompt)
            
            if result['success']:
                print(f"✅ Генерация успешна!")
                print(f"📁 Файл: {result['image_path']}")
                print(f"📊 Метаданные: {result['metadata']}")
            else:
                print(f"❌ Ошибка генерации: {result['error']}")
                
        except Exception as e:
            print(f"❌ Исключение при тестировании: {e}")
    
    # Тестируем ленивую инициализацию других провайдеров
    print(f"\n🧪 Тестирование ленивой инициализации...")
    
    for provider_id, provider in manager.providers.items():
        if provider_id == 'mock':
            continue  # Mock уже протестирован
            
        print(f"\n🔄 Тестирование ленивой инициализации {provider.get_name()}...")
        
        # Проверяем статус до инициализации
        status_before = manager.get_provider_status()[provider_id]
        print(f"Статус до инициализации: {status_before['status']}")
        
        # Пытаемся инициализировать
        try:
            success = await manager.initialize_provider_lazy(provider_id)
            print(f"Результат инициализации: {'✅ Успешно' if success else '❌ Неудачно'}")
            
            # Проверяем статус после инициализации
            status_after = manager.get_provider_status()[provider_id]
            print(f"Статус после инициализации: {status_after['status']}")
            
        except Exception as e:
            print(f"❌ Исключение при инициализации: {e}")
    
    print("\n🏁 Тестирование завершено!")

async def test_mock_provider_only():
    """Тестирует только mock провайдер"""
    print("🧪 Тестирование Mock провайдера...")
    
    # Создаем mock провайдер
    mock_provider = MockProvider()
    
    # Инициализируем
    success = await mock_provider.initialize()
    print(f"Инициализация: {'✅' if success else '❌'}")
    
    if success:
        # Тестируем генерацию
        test_prompts = [
            "beautiful sunset over ocean",
            "cute cat sitting on a windowsill",
            "futuristic city with flying cars",
            "abstract art with vibrant colors"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n🎨 Тест {i}: {prompt}")
            
            result = await mock_provider.generate_image(
                prompt,
                width=512,
                height=512
            )
            
            if result['success']:
                print(f"✅ Изображение создано: {result['image_path']}")
            else:
                print(f"❌ Ошибка: {result['error']}")
    
    print("\n🏁 Тестирование Mock провайдера завершено!")

def main():
    """Главная функция"""
    print("🤖 Тестирование Telegram бота для генерации изображений")
    print("=" * 60)
    
    # Загружаем переменные окружения
    load_dotenv()
    
    # Проверяем наличие токенов
    hf_token = os.getenv("HF_TOKEN")
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    print(f"🔑 HF_TOKEN: {'✅' if hf_token else '❌'}")
    print(f"🔑 TELEGRAM_BOT_TOKEN: {'✅' if telegram_token else '❌'}")
    
    if not hf_token:
        print("\n⚠️ HF_TOKEN не найден. Будут протестированы только mock провайдеры.")
        asyncio.run(test_mock_provider_only())
    else:
        print("\n🚀 Запуск полного тестирования...")
        asyncio.run(test_providers())

if __name__ == "__main__":
    main()
