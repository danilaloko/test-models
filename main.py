#!/usr/bin/env python3
"""
Главный скрипт для управления генерацией изображений
"""

import os
import sys
from dotenv import load_dotenv

def check_environment():
    """Проверяем настройки окружения"""
    load_dotenv()
    
    if not os.environ.get("HF_TOKEN"):
        print("❌ Ошибка: HF_TOKEN не найден в .env файле")
        print("📝 Создайте .env файл на основе env.example и добавьте ваш токен")
        return False
    
    return True

def show_menu():
    """Показываем меню выбора"""
    print("\n🎨 Генератор изображений с помощью AI")
    print("=" * 40)
    print("1. Генерация через Hugging Face API (рекомендуется)")
    print("2. Локальная генерация (требует GPU)")
    print("3. Тест NSFW модели через API")
    print("4. Показать информацию о проекте")
    print("5. Выход")
    print("=" * 40)

def run_hf_api():
    """Запускаем генерацию через Hugging Face API"""
    print("\n🚀 Запуск генерации через Hugging Face API...")
    os.system("python test-hf-api.py")

def run_local():
    """Запускаем локальную генерацию"""
    print("\n🖥️  Запуск локальной генерации...")
    print("⚠️  Внимание: это может занять много времени и ресурсов")
    os.system("python test-nsfw.py")

def run_nsfw_api():
    """Запускаем NSFW модель через API"""
    print("\n🔞 Запуск NSFW модели через API...")
    os.system("python test-nsfw-api.py")

def show_info():
    """Показываем информацию о проекте"""
    print("\n📋 Информация о проекте:")
    print("- Проект: test-models")
    print("- Описание: Генерация изображений с помощью AI")
    print("- Поддерживаемые модели: Stable Diffusion, NSFW модели")
    print("- Способы генерации: Hugging Face API, локально")
    print("\n📁 Структура:")
    print("- generated_images/ - сгенерированные изображения")
    print("- models/ - локальные модели")
    print("- .env - переменные окружения")
    print("\n🔗 Полезные ссылки:")
    print("- Hugging Face токены: https://huggingface.co/settings/tokens")
    print("- Документация: README.md")

def main():
    """Главная функция"""
    if not check_environment():
        return
    
    while True:
        show_menu()
        choice = input("\nВыберите опцию (1-5): ").strip()
        
        if choice == "1":
            run_hf_api()
        elif choice == "2":
            run_local()
        elif choice == "3":
            run_nsfw_api()
        elif choice == "4":
            show_info()
        elif choice == "5":
            print("\n👋 До свидания!")
            break
        else:
            print("\n❌ Неверный выбор. Попробуйте снова.")
        
        input("\nНажмите Enter для продолжения...")

if __name__ == "__main__":
    main()
