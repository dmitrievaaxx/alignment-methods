Тебе нужно: 
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

2) Установить зависимости для verl
   - pip install torch torchvision torchaudio transformers datasets accelerate bitsandbytes
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

**Замечания:**    
Я в C:\Users\darya\Documents\alignment-methods\verl\verl\trainer создала файл под названием sft_trainer_minimal.yaml для тестового запуска
