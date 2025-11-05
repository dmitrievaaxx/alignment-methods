# clearml_setup.py
import os
from clearml import Task

def setup_clearml():
    try:
        task = Task.init(
            project_name="alignment-methods",
            task_name="sft_1.5b_12gb_offline_wandb",
            auto_connect_frameworks={
                'pytorch': True,
                'tensorflow': False,
                'tensorboard': True,
                'matplotlib': True,
                'xgboost': False,
                'scikit': True,
                'fastai': False,
                'lightgbm': False,
                'hydra': True,
                'detectron2': False,
                'transformers': True,
                'jsonargparse': True
            }
        )
        
        # Мгновенная отправка метрик
        task.get_logger().set_reporting(enable_immediate_reporting=True)
        
        # Сохраняем конфигурацию
        config_path = "/job/verl_config/sft_qwen_1.5b.yaml"
        if os.path.exists(config_path):
            task.connect_configuration(config_path, name="training_config")
            print("Config saved to ClearML")
        
        task.logger.report_text("Starting training")
        
        print(f"ClearML started: {task.id}")
        return task
        
    except Exception as e:
        print(f"ClearML failed: {e}")
        return None

if __name__ == "__main__":
    setup_clearml()
