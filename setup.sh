#!/bin/bash

# Скрипт для настройки проекта test-models

echo "🚀 Настройка проекта test-models..."

# Проверяем наличие Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден. Установите Python 3.8+ и попробуйте снова."
    exit 1
fi

# Создаем виртуальное окружение если его нет
if [ ! -d "env" ]; then
    echo "📦 Создание виртуального окружения..."
    python3 -m venv env
fi

# Активируем виртуальное окружение
echo "🔧 Активация виртуального окружения..."
source env/bin/activate

# Устанавливаем зависимости
echo "📚 Установка зависимостей..."
pip install --upgrade pip
pip install -r requirements.txt

# Создаем .env файл если его нет
if [ ! -f ".env" ]; then
    echo "⚙️  Создание .env файла..."
    cp env.example .env
    echo "📝 Отредактируйте .env файл и добавьте ваш HF_TOKEN"
fi

# Создаем необходимые папки
echo "📁 Создание папок..."
mkdir -p generated_images models

echo "✅ Настройка завершена!"
echo ""
echo "📋 Следующие шаги:"
echo "1. Отредактируйте .env файл и добавьте ваш HF_TOKEN"
echo "2. Активируйте виртуальное окружение: source env/bin/activate"
echo "3. Запустите тест: python test-hf-api.py"
echo ""
echo "🎯 Для получения токена: https://huggingface.co/settings/tokens"
