import re
import torch
import PyPDF2
from TTS.api import TTS
from playsound import playsound
from pydub import AudioSegment
import os

# Verifica se a GPU está disponível e define o dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# Lista os modelos disponíveis de TTS
# print(TTS().list_models())

# Inicializa o TTS com o modelo específico para português, ajustando para usar a GPU se disponível
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)

# Define o nome do arquivo e o arquivo de áudio do locutor
nome_arquivo = "Ferramentas dos Titãs - Tim Ferriss.pdf"
arquivo_locutor = "sons/bruna_ambiente.wav"

try:
    # Usa o gerenciador de contexto para abrir o arquivo
    with open(nome_arquivo, 'rb') as livro:
        pdf = PyPDF2.PdfReader(livro)

        numero_paginas = len(pdf.pages)
        print(f"Total de páginas: {numero_paginas}")

        # Inicializa uma string vazia para armazenar o texto
        texto_completo = ""

        # Itera sobre cada página do PDF e extrai o texto
        for pagina_num in range(numero_paginas):
            pagina = pdf.pages[pagina_num]
            texto_pagina = pagina.extract_text()
            texto_completo += texto_pagina + " "

        # Remove espaços extras e quebras de linha do texto
        texto_limpo = texto_limpo = re.sub(r'[©""*_/\s]+', ' ', texto_completo).strip().replace('.', '!').replace('\n', ' ')

        print("Texto ----------->", texto_limpo)
        # Salva o texto limpo em um arquivo .txt
        with open("output.txt", "w", encoding="utf-8") as arquivo:
            arquivo.write(texto_limpo)

    # Cria a pasta "audios" se não existir
    pasta_audios = "audios"
    if not os.path.exists(pasta_audios):
        os.makedirs(pasta_audios)

    # Divide o texto em pedaços menores
    tamanho_pedaco = 203
    pedacos = [texto_limpo[i:i+tamanho_pedaco] for i in range(0, len(texto_limpo), tamanho_pedaco)]

    # Cria um objeto AudioSegment vazio para combinar os áudios
    combined_audio = AudioSegment.empty()

    # Gera fala a partir de cada pedaço de texto em português e salva em um arquivo na pasta "audios"
    for i, pedaco in enumerate(pedacos):
        caminho_arquivo = os.path.join(pasta_audios, f"output_{i}.wav")
        tts.tts_to_file(text=pedaco,
                        file_path=caminho_arquivo,
                        speaker_wav=arquivo_locutor,
                        language="pt")

        # Carrega o áudio gerado e o adiciona ao áudio combinado
        audio = AudioSegment.from_wav(caminho_arquivo)
        combined_audio += audio

    # Salva o áudio combinado em um arquivo
    combined_audio.export("output_completo.wav", format="wav")

    # Toca o áudio combinado
    playsound("output_completo.wav")

except Exception as e:
    print(f"Ocorreu um erro: {e}")