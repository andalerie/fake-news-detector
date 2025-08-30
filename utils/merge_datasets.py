import os
import pandas as pd

def load_pair(folder, fake_name='Fake.csv', true_name='True.csv'):
    # пути к файлам
    fake_path = os.path.join(folder, fake_name)
    true_path = os.path.join(folder, true_name)

    # читаем CSV
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    # если в файлах колонки 'title' и 'text', объединяем их в единый текст
    if 'title' in df_fake.columns and 'text' in df_fake.columns:
        df_fake['text'] = df_fake['title'] + '. ' + df_fake['text']
        df_true['text'] = df_true['title'] + '. ' + df_true['text']

    # добавляем метки: фейк=1, правда=0
    df_fake['label'] = 1
    df_true['label'] = 0

    # оставляем только нужные столбцы
    return pd.concat([
        df_true[['text','label']],
        df_fake[['text','label']]
    ], ignore_index=True)

def main():
    # пути к исходным датасетам
    base = 'data/raw'
    df1 = load_pair(os.path.join(base, 'first_dataset'))
    df2 = load_pair(os.path.join(base, 'second_dataset'))

    # объединяем и перемешиваем
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # сохраняем итог
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/combined_news.csv', index=False)
    print(f"✔ Объединено {len(df)} статей: real={int((df.label==0).sum())}, fake={int((df.label==1).sum())}")
    print("✔ Сохранено в data/combined_news.csv")

if __name__ == '__main__':
    main()
