import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import sys

def load_model():
    """Загружает модель и токенизатор с правильными параметрами"""
    try:
        print("Загружаем модель NSFW-3B...")
        
        # Пробуем различные варианты загрузки для исправления ошибки с тензорами
        load_configs = [
            # Конфигурация 1: Без device_map, CPU сначала
            {
                "trust_remote_code": True,
                "torch_dtype": torch.float32,
                "low_cpu_mem_usage": True,
                "use_safetensors": False
            },
            # Конфигурация 2: float16 без device_map
            {
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
                "use_safetensors": False
            },
            # Конфигурация 3: Принудительная загрузка с исправлением конфигурации
            {
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "low_cpu_mem_usage": True,
                "use_safetensors": False,
                "attn_implementation": "eager"  # Принудительно используем eager attention
            }
        ]
        
        model = None
        for i, config in enumerate(load_configs):
            try:
                print(f"Попытка загрузки {i+1}/3...")
                model = AutoModelForCausalLM.from_pretrained(
                    "UnfilteredAI/NSFW-3B",
                    **config
                )
                
                # Если device_map не указан, перемещаем на CUDA вручную
                if "device_map" not in config and torch.cuda.is_available():
                    model = model.to("cuda")
                
                print(f"✅ Модель загружена успешно (конфигурация {i+1})")
                break
                
            except Exception as e:
                print(f"❌ Конфигурация {i+1} не сработала: {e}")
                if i == len(load_configs) - 1:
                    print("Все попытки загрузки провалились")
                    return None, None
                continue
        
        # Загружаем токенизатор
        tokenizer = AutoTokenizer.from_pretrained(
            "UnfilteredAI/NSFW-3B", 
            trust_remote_code=True
        )
        
        # Устанавливаем pad_token если его нет
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Проверяем совместимость модели
        try:
            print("Тестируем модель...")
            test_input = tokenizer("Тест", return_tensors="pt")
            if torch.cuda.is_available() and model.device.type == 'cuda':
                test_input = {k: v.to(model.device) for k, v in test_input.items()}
            
            with torch.no_grad():
                _ = model(**test_input)
            print("✅ Модель прошла тест совместимости")
            
        except Exception as e:
            print(f"❌ Модель не прошла тест совместимости: {e}")
            return None, None
            
        return model, tokenizer
        
    except Exception as e:
        print(f"Критическая ошибка при загрузке модели: {e}")
        return None, None

def create_prompt(system_message, user_message):
    """Создает промпт в правильном формате"""
    return f"""<|im_start|>system
{system_message}
<|im_end|>
<|im_start|>user
{user_message}
<|im_end|>
<|im_start|>assistant
"""

def generate_response(model, tokenizer, prompt, max_length=512):
    """Генерирует ответ модели с улучшенной обработкой ошибок"""
    try:
        # Токенизируем входной текст с ограничением длины
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024,  # Уменьшаем максимальную длину
            padding=False
        )
        
        # Перемещаем на GPU если доступно
        if torch.cuda.is_available() and model.device.type == 'cuda':
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        print(f"Размер входных данных: {inputs['input_ids'].shape}")
        
        # Создаем стример для вывода (но без него сначала для отладки)
        # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Попытки генерации с разными параметрами
        generation_configs = [
            # Конфигурация 1: Консервативная
            {
                "max_new_tokens": min(max_length, 256),
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": False,  # Отключаем кеш для избежания проблем
                "repetition_penalty": 1.1
            },
            # Конфигурация 2: Жадный поиск
            {
                "max_new_tokens": min(max_length, 128),
                "do_sample": False,  # Жадный поиск
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": False
            },
            # Конфигурация 3: Минимальная
            {
                "max_new_tokens": 50,
                "temperature": 1.0,
                "do_sample": True,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": False
            }
        ]
        
        for i, config in enumerate(generation_configs):
            try:
                print(f"Попытка генерации {i+1}/3 с параметрами: max_new_tokens={config['max_new_tokens']}")
                
                # Генерируем ответ
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        **config
                    )
                
                # Декодируем ответ
                generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                if response.strip():  # Проверяем, что ответ не пустой
                    print(f"✅ Генерация успешна (конфигурация {i+1})")
                    return response.strip()
                else:
                    print(f"⚠️ Конфигурация {i+1} дала пустой ответ")
                    continue
                    
            except Exception as e:
                print(f"❌ Конфигурация генерации {i+1} не сработала: {e}")
                if i == len(generation_configs) - 1:
                    return "Извините, не удалось сгенерировать ответ после всех попыток."
                continue
        
        return "Извините, все попытки генерации завершились неудачей."
        
    except Exception as e:
        print(f"Критическая ошибка при генерации: {e}")
        return f"Произошла критическая ошибка: {str(e)}"

def main():
    """Основная функция для непрерывного общения с моделью"""
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        print("Не удалось загрузить модель. Завершение работы.")
        return
    
    print("\n" + "="*50)
    print("🤖 Чат с NSFW-3B запущен!")
    print("Введите 'quit', 'exit' или 'выход' для завершения")
    print("Введите 'clear' для очистки истории")
    print("="*50 + "\n")
    
    # Системное сообщение
    system_message = "You are a helpful AI assistant."
    
    while True:
        try:
            # Получаем ввод пользователя
            user_input = input("\n👤 Вы: ").strip()
            
            # Проверяем команды выхода
            if user_input.lower() in ['quit', 'exit', 'выход', 'q']:
                print("👋 До свидания!")
                break
            
            # Проверяем команду очистки
            if user_input.lower() in ['clear', 'очистить']:
                print("🧹 История очищена!")
                continue
            
            # Пропускаем пустые сообщения
            if not user_input:
                continue
            
            # Создаем промпт
            prompt = create_prompt(system_message, user_input)
            
            print("\n🤖 Ассистент: ", end="", flush=True)
            
            # Генерируем ответ
            response = generate_response(model, tokenizer, prompt)
            
            print()  # Новая строка после ответа
            
        except KeyboardInterrupt:
            print("\n\n👋 До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Произошла ошибка: {e}")
            continue

if __name__ == "__main__":
    main()
