from openai import OpenAI

client = OpenAI(
    api_key="sk-aitunnel-CBUNVMMU6Kcw1xyswONaTxiqpcseUeux", # Ключ из нашего сервиса
    base_url="https://api.aitunnel.ru/v1/",
)

chat_result = client.chat.completions.create(
    messages=[{"role": "user", "content": "Найди этот bar code и пришли нвазвание товара 8809576261752"}],
    model="sonar",
    max_tokens=50000, # Старайтесь указывать для более точного расчёта цены
)
print(chat_result.choices[0].message)