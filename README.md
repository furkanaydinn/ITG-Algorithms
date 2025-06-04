# Zaman Aralıklı Graf (ITG) Kısa Yol Algoritmaları

Bu projede, zaman aralıklı (interval time graph, ITG) üzerinde baskın olmayan yolları ve en kısa yolları bulmak için hem Rust hem de Python ile sıralı ve paralel algoritmaların karşılaştırmalı uygulamaları yer almaktadır.

## Klasörler ve Dosyalar

- `paralelism/` : Rust ile sıralı ve Rayon tabanlı paralel ITG algoritması.
- `ITG_python.py` : Python ile sıralı ve multiprocessing tabanlı paralel ITG algoritması.

## Kullanım

### Rust (paralelism)

1. Gerekli bağımlılıkları yükleyin:
   ```sh
   cd paralelism
   cargo build --release
   ```
2. Programı çalıştırın:
   ```sh
   cargo run --release
   ```
3. Program, büyük bir sentetik ITG grafı üzerinde hem sıralı hem de paralel algoritmanın çalışma süresini ve örnek sonuçlarını ekrana yazacaktır.

### Python (ITG_python.py)

1. Gerekli paketler: Python 3, multiprocessing (standart kütüphane)
2. Programı çalıştırın:
   ```sh
   python3 ITG_python.py
   ```
3. Program, büyük bir sentetik ITG grafı üzerinde hem sıralı hem de paralel algoritmanın çalışma süresini ve örnek sonuçlarını ekrana yazacaktır.

## Açıklama

- **manage_dominance_and_sort**: Hem Rust hem Python'da, baskın olmayan yolları bulur ve sıralar.
- **generate_deterministic_itg**: Rastgele ama tekrarlanabilir (seed ile) ITG grafı üretir.
- **itg_shortest_path_algorithm2_rust_sequential / python_sequential**: Sıralı kısa yol algoritması.
- **itg_shortest_path_algorithm2_rust_parallel / python_multiprocessing**: Paralel kısa yol algoritması.

## Notlar
- Rust kodunda paralelleştirme için [rayon](https://crates.io/crates/rayon) kullanılır.
- Python kodunda paralelleştirme için `multiprocessing` modülü kullanılır.
- Her iki dilde de algoritmaların çıktıları ve süreleri karşılaştırılabilir.

## Lisans
MIT
