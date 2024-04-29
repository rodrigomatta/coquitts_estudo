import torch
from TTS.api import TTS
from playsound import playsound  # Import playsound to play the generated audio file

# Verifica se a GPU está disponível e define o dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# Lista os modelos disponíveis de TTS
print(TTS().list_models())

# Inicializa o TTS com o modelo específico para português, ajustando para usar a GPU se disponível
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Gera fala a partir de um texto em português e salva em um arquivo
tts.tts_to_file(text="Verificar a instalação do RealtimeTTS: Como mencionado anteriormente, o erro ModuleNotFoundError: No module named 'RealtimeTTS' indica que o módulo RealtimeTTS não está instalado. No entanto, parece haver um mal-entendido, pois RealtimeTTS não é um pacote Python padrão e pode não estar disponível para instalação via pip. Se você está tentando usar uma biblioteca específica para TTS em tempo real, certifique-se de que está utilizando o nome correto do pacote e que ele está disponível para instalação. Caso contrário, você pode precisar procurar uma alternativa ou implementar a funcionalidade de TTS em tempo real de outra forma.",
                 file_path="output.wav",
                 speaker_wav="rochele.wav",  # Caminho para um arquivo de áudio de voz alvo
                 language="pt")  # Especifica o idioma como português do Brasil

# Toca o arquivo de áudio gerado
playsound("output.wav")
