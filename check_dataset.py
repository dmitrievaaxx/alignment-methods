# check_dataset.py
import pandas as pd

def check_dataset():
    print("Проверка датасета...")
    
    # Проверяем train
    train_df = pd.read_parquet("data/train.parquet")
    print(f"Train samples: {len(train_df)}")
    print("Первые 3 примера:")
    for i in range(min(3, len(train_df))):
        print(f"\n--- Пример {i+1} ---")
        print(f"Input: {train_df['input'].iloc[i][:200]}...")
        print(f"Output: {train_df['output'].iloc[i][:200]}...")
    
    # Проверяем val
    val_df = pd.read_parquet("data/val.parquet")
    print(f"\nVal samples: {len(val_df)}")
    
    # Проверяем наличие нужных колонок
    assert 'input' in train_df.columns, "Нет колонки 'input' в train"
    assert 'output' in train_df.columns, "Нет колонки 'output' в train"
    assert 'input' in val_df.columns, "Нет колонки 'input' в val"
    assert 'output' in val_df.columns, "Нет колонки 'output' в val"
    
    print("✅ Дадасет в правильном формате!")

if __name__ == "__main__":
    check_dataset()