# disable_flash_attn.py
import os
import sys

# ЖЕСТКО отключаем flash attention ДО ЛЮБЫХ импортов
os.environ['USE_FLASH_ATTENTION'] = '0'
os.environ['TRANSFORMERS_USE_FLASH_ATTENTION'] = '0' 
os.environ['DISABLE_FLASH_ATTENTION'] = '1'
os.environ['FORCE_ATTENTION_IMPLEMENTATION'] = 'eager'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Для полного отключения
os.environ['FLASH_ATTENTION_SKIP'] = '1'
os.environ['FLASH_ATTENTION_ALWAYS_DISABLE'] = '1'

print("✓ Flash Attention переменные окружения установлены")

# Импортируем и патчим
import transformers

# Агрессивный патчинг
def disable_flash_attn():
    # Патчим все возможные функции проверки
    transformers.utils.import_utils.is_flash_attn_2_available = lambda: False
    transformers.utils.import_utils.is_flash_attn_greater_or_equal_2_10 = lambda: False  
    transformers.utils.import_utils.can_use_flash_attention_2 = lambda *args, **kwargs: False
    
    # Патчим для разных версий transformers
    if hasattr(transformers.utils.import_utils, 'is_flash_attn_available'):
        transformers.utils.import_utils.is_flash_attn_available = lambda: False
    
    # Патчим конфигурацию моделей
    try:
        from transformers import PretrainedConfig
        original_from_pretrained = PretrainedConfig.from_pretrained
        
        @classmethod
        def patched_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            # Принудительно отключаем flash attention в конфиге
            kwargs['attn_implementation'] = 'eager'
            result = original_from_pretrained(pretrained_model_name_or_path, **kwargs)
            if hasattr(result, 'attn_implementation'):
                result.attn_implementation = 'eager'
            return result
            
        PretrainedConfig.from_pretrained = patched_from_pretrained
        print("✓ Патч PretrainedConfig применен")
    except Exception as e:
        print(f"⚠ Ошибка патчинга конфига: {e}")

disable_flash_attn()
print("✓ Flash Attention полностью отключен")
