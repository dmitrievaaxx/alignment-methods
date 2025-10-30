## Для переноса Yandex.Datashere в консоль (локально нужно сделать)
1. pip install datasphere 
2. irm https://storage.yandexcloud.net/yandexcloud-yc/install.ps1 | iex
3. yc init
y0__xClps-ABBjB3RMggLSY-BQwpf2rsgiq1qZpFf1mc81i6jZmzqwVjcJTsg
4. datasphere project list -c bt16n0d8m6jfrc9rv4vs (будет id нашего проекта в сообществе)
5. datasphere project job execute -p bt1jvegm7p69m5a6rnoa -c config.yaml  (проверка, что все корректно)

## Установка verl: (уже настроено)
6. pip install torch torchvision torchaudio transformers datasets accelerate bitsandbytes (надо установить думаю)
7. git clone https://github.com/volcengine/verl.git (сделано)
8. cd verl (сделано)
9. git checkout 8b33abd84f360473f05e5a750aef36e974340cce (сделано)
10. Изменить в setup.py: long_description = (this_directory / "README.md").read_text(encoding="utf-8") (сделано)
11. pip install -e . (сделано)
12. Проверка python check_verl.py

##Тестовый запуск:
- Создать файл в C:\Users\darya\Documents\alignment-methods\verl\verl\trainer под названием sft_trainer_minimal.yaml (создано)
- Copy-Item data\train.parquet train_data.parquet (сделано)
  Copy-Item data\val.parquet val_data.parquet (сделано)
- По идее: datasphere project job execute -p bt1jvegm7p69m5a6rnoa -c config.yaml (требует Линукс)
