## 31.10.2025

1) Перенести Yandex.Datasphere в консоль:
   - pip install datasphere
   - irm https://storage.yandexcloud.net/yandexcloud-yc/install.ps1 | iex
   - yc init и там ввести (y0__xClps-ABBjB3RMggLSY-BQwpf2rsgiq1qZpFf1mc81i6jZmzqwVjcJTsg)
   - datasphere project list -c bt16n0d8m6jfrc9rv4vs (будет id нашего проекта в сообществе)
  
Проверка:
```
PS C:\Users\darya\Documents\alignment-methods> datasphere project job execute -p bt1jvegm7p69m5a6rnoa -c basic_config.yaml
2025-10-31 01:19:04,267 - [INFO] - logs file path: C:\Users\darya\AppData\Local\Temp\datasphere\job_2025-10-31T01-19-04.251014
2025-10-31 01:19:42,437 - [INFO] - creating job ...
2025-10-31 01:19:43,865 - [INFO] - no files to upload
2025-10-31 01:19:43,865 - [INFO] - created job `bt1sna53qma774d9heh6`
2025-10-31 01:19:45,343 - [INFO] - executing job ...
2025-10-31 01:20:37,400 - Hello from DataSphere!
2025-10-31 01:20:37,400 - Arguments: ['--data', '/job/data_7l2', '--result', '/job/output_dp0']
2025-10-31 01:21:27,842 - [INFO] - job link: https://datasphere.yandex.cloud/communities/bt16n0d8m6jfrc9rv4vs/projects/bt1jvegm7p69m5a6rnoa/job/bt1sna53qma774d9heh6
2025-10-31 01:21:27,843 - [INFO] - downloading 1 files (123.0B) ...
2025-10-31 01:21:28,639 - [INFO] - files are downloaded
2025-10-31 01:21:28,639 - [INFO] - job completed successfully
```
2) Verl
   - pip install torch torchvision torchaudio transformers datasets accelerate bitsandbytes
   - git clone https://github.com/volcengine/verl.git
   - cd verl
   - git checkout 8b33abd84f360473f05e5a750aef36e974340cce
   - Изменить в setup.py: long_description = (this_directory / "README.md").read_text(encoding="utf-8")
   - pip install -e .
  
Проверка: 
```
PS C:\Users\darya\Documents\alignment-methods> python check_verl.py
✅ VERL импортировалась успешно!
Версия PyTorch: 2.8.0+cpu
Версия Transformers: 4.57.1
✅ Основные модули verl доступны.
```

3) Тестовый запуск
   ```
   datasphere project job execute -p bt1jvegm7p69m5a6rnoa -c config.yaml
   ```
   или
   ```
   python -m datasphere.main project job execute -p bt1jvegm7p69m5a6rnoa -c config.yaml
   ```

---
## 03.11.2025  
`config.yaml` в корне проекта 
```yaml
# config.yaml
name: test-job-dmitrieva
desc: Simple working version

cmd: |
  export PIP_ROOT_USER_ACTION=ignore
  export USE_FLASH_ATTENTION=0
  export TRANSFORMERS_USE_FLASH_ATTENTION=0
  pip install --upgrade pip
  pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
  pip install packaging ninja wheel setuptools psutil
  pip install flash-attn>=2.5.0 --no-build-isolation --no-cache-dir
  pip install accelerate transformers datasets peft wandb hydra-core omegaconf tensordict ray codetiming
  mkdir -p /job
  cp -r ${VERL_DIR} /job/verl
  mkdir -p /job/verl_config
  cp /job/verl/verl/trainer/sft_qwen_1.5b.yaml /job/verl_config/
  python ${UPDATE_CONFIG} /job/verl_config sft_qwen_1.5b
  cd /job/verl && PYTHONPATH=/job/verl:$PYTHONPATH python ${DISABLE_FLASH_ATTN} && PYTHONPATH=/job/verl:$PYTHONPATH torchrun --standalone --nnodes=1 --nproc_per_node=1 verl/trainer/fsdp_sft_trainer.py --config-path /job/verl_config --config-name sft_qwen_1.5b data.train_files="[${TRAIN_DATA}]" data.val_files="[${VAL_DATA}]"

inputs:
  - train_data.parquet: TRAIN_DATA
  - val_data.parquet: VAL_DATA
  - update_config.py: UPDATE_CONFIG
  - verl: VERL_DIR
  - disable_flash_attn.py: DISABLE_FLASH_ATTN

outputs:
  - output_dir: OUTPUT_DIR

cloud-instance-type: g1.1
```

Cоздать файл `disable_flash_attn.py` в корне проекта
```python
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
```

`sft_qwen_1.5b.yaml` в verl/verl/trainer
```yaml
# sft_qwen_1.5b.yaml
trainer:
  project_name: gsm8k_qwen_sft
  experiment_name: sft_1.5b_12gb
  logger: ['console']  
  default_local_dir: ./checkpoints
  total_epochs: 1
  save_freq: 500
  eval_freq: 100
  test_freq: 100
  total_training_steps: 1000
  seed: 42
  nnodes: 1
  n_gpus_per_node: 1
  save_total_limit: 2
  logging_steps: 10

model:
  partial_pretrain: "Qwen/Qwen2.5-1.5B"
  trust_remote_code: true
  strategy: fsdp
  enable_gradient_checkpointing: true
  lora_rank: 8
  lora_alpha: 16
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  use_flash_attention: false
  attn_implementation: "eager"
  dtype: "bfloat16" 
  fsdp_config:
    sharding_strategy: "NO_SHARD"
    mixed_precision: "bf16"
    use_orig_params: true

optim:
  lr: 2e-5
  betas: [0.9, 0.95]
  weight_decay: 0.01
  warmup_steps_ratio: 0.03
  lr_scheduler: cosine
  clip_grad: 1.0

data:
  train_files: [] 
  val_files: []
  global_batch_size: 32
  micro_batch_size_per_gpu: 2
  grad_accumulation_steps: 16
  max_length: 1024
  prompt_key: input
  response_key: output
  balance_dp_token: true
  num_workers: 4
  pin_memory: true
```

**Главное:** исправить файл `fsdp_sft_trainer.py` в verl/verl/trainer
В самом начале файла указать:  
```python
import os
os.environ['USE_FLASH_ATTENTION'] = '0'
os.environ['TRANSFORMERS_USE_FLASH_ATTENTION'] = '0'
os.environ['FORCE_ATTENTION_IMPLEMENTATION'] = 'eager'
```
Найти строчку self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(...) и в ней исправить attn_implementation="eager" и dtype:  
```python
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                dtype=torch_dtype,
                attn_implementation="eager",
                trust_remote_code=trust_remote_code,
            )
```


## 05.11.25  
1. ✅ wandb в offline режиме

```python
# sft_qwen_1.5b.yaml
trainer:
  project_name: gsm8k_qwen_sft
  experiment_name: sft_1.5b_12gb
  logger: ['console', 'wandb']  
  default_local_dir: ./checkpoints
  wandb_dir: ./wandb 
  wandb_mode: "offline"
  total_epochs: 1
  save_freq: 500
  eval_freq: 100
  test_freq: 100
  total_training_steps: 1000
  seed: 42
  nnodes: 1
  n_gpus_per_node: 1
  save_total_limit: 2
  logging_steps: 10
```

В `config.yaml` (измененная версия запушена в репозиторий) добавлено: 
- `export WANDB_MODE=offline` после строчки export `TRANSFORMERS_USE_FLASH_ATTENTION=0`
- `pip install wandb --upgrade` после строчки `pip install accelerate transformers datasets peft wandb hydra-core omegaconf tensordict ray codetiming`

2. Для устранения предупреждения в файле `fsdp_sft_trainer.py` исправлены все `FULL_SHARD` на `NO_SHARD`
3. ✅ ClearML online
   
```python
api {
  # Dasha Dmitrieva's workspace
  web_server: https://app.clear.ml/
  api_server: https://api.clear.ml
  files_server: https://files.clear.ml
  # alighment-methods
  credentials {
    "access_key" = "1115V427IJEV3GB0UZMCHFD8XPCLO9"
    "secret_key" = "3k9gBL0lsd9iKBzHvaUpjwXOYtgI7HObE9De98qMDnrBQ0WZQrwITY9Q2PcR4kWWmPs"
  }
}
```
- Поэтому добавляем в `config.yaml` (запушено в репозиторий):  
  
После `export WANDB_MODE=offline` добавить:  
```yaml
  export CLEARML_API_HOST=https://api.clear.ml
  export CLEARML_WEB_HOST=https://app.clear.ml  
  export CLEARML_FILES_HOST=https://files.clear.ml
  export CLEARML_API_ACCESS_KEY="1115V427IJEV3GB0UZMCHFD8XPCLO9"
  export CLEARML_API_SECRET_KEY="3k9gBL0lsd9iKBzHvaUpjwXOYtgI7HObE9De98qMDnrBQ0WZQrwITY9Q2PcR4kWWmPs"
```
После `pip install wandb --upgrade` добавить:  
```yaml
pip install clearml
```

- Изменения в файле `fsdp_sft_trainer.py`:
  1. В самом начале файла добавить:
```python
import os
try:
    from clearml import Task
    task = Task.init(
        project_name="alignment-methods",
        task_name="sft_1.5b_12gb_offline_wandb", 
        auto_connect_frameworks=True
    )
    print(f"🎯 ClearML INITIALIZED: {task.id}")
except Exception as e:
    print(f"💥 ClearML INIT FAILED: {e}")
```
  2. В методе `init` класса FSDPSFTTrainer добавить строку: `self.global_step = 0`  
  3. В методе `training_step` добавить перед return:  
```python
        try:
            from clearml import Task
            task = Task.current_task()
            print(f"🔍 ClearML debug: task={task}, has_logger={task and task.logger}")
            if task and task.logger:
                print(f"📊 ClearML logging: loss={step_loss.detach().item()}, lr={lr}, step={self.global_step}")
                task.logger.report_scalar("train", "loss", step_loss.detach().item(), iteration=self.global_step)
                task.logger.report_scalar("train", "lr", lr, iteration=self.global_step)
                print("✅ ClearML metrics sent successfully")
            else:
                print("❌ ClearML: no task or logger")
        except Exception as e:
            print(f"❌ ClearML error: {e}")

        self.global_step += 1
```
  4. В методе `validation_step` добавить перед return:  
```python
        try:
            from clearml import Task
            task = Task.current_task()
            if task and task.logger:
                task.logger.report_scalar("validation", "loss", loss.item(), iteration=self.global_step)
        except:
            pass
```
