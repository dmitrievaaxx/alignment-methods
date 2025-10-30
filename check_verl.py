# check_verl.py

try:
    import torch
    import transformers
    import verl

    print("✅ VERL импортировалась успешно!")
    print("Версия PyTorch:", torch.__version__)
    print("Версия Transformers:", transformers.__version__)

    # Проверим, что ключевые модули verl доступны
    # from verl.training import Trainer
    # from verl.utils import logging
    from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
    from verl.utils import logging_utils as logging


    print("✅ Основные модули verl доступны.")
except Exception as e:
    print("❌ Ошибка при импорте:", e)