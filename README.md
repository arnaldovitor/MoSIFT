# Motion SIFT (MoSIFT)
Implementação do artigo MoSIFT: Recognizing Human Actions in Surveillance Videos.

A. feature_extractor.py
> Módulo de extração de característica. Recebe vídeos e tem como retorno uma lista de características MoSIFT (concatenação do descritor  SIFT com descritor HoF).

B. vbow.py
> Módulo de Visual Bag of Words. Recebe os feature vectors obtidos pelo extrator MoSIFT e retorna uma lista de histogramas afim de serem utilizados posteriormente em um classificador.

C. util.py
> Módulo de utilidades. Possui funções utéis para o funcionamento geral do método.