# simple-llm-c

Implementação de um LLM (Large Language Model) do zero em C puro, construído progressivamente em 6 fases didáticas.

## Estrutura do Projeto

```
simple-llm-c/
├── phase1/    — Álgebra linear, softmax, backpropagation
├── phase2/    — Tokenizador BPE (Byte-Pair Encoding)
├── phase3/    — Word2Vec (skip-gram, embeddings densos)
├── phase4/    — FFLM (Feed-Forward Language Model)
├── phase5/    — Transformer decoder-only (GPT-style)
├── phase6/    — Quantização INT8/INT4
├── pipeline/  — Chat interativo integrado
└── generate_chat_corpus.py
```

## Fases

### Fase 1 — Fundamentos Matemáticos
Operações de álgebra linear em C: multiplicação de matrizes, softmax, funções de ativação (ReLU, tanh, sigmoid), backpropagation manual, otimizador SGD.

### Fase 2 — Tokenizador BPE
Byte-Pair Encoding do zero: contagem de pares, merge iterativo, vocabulário de subpalavras, encode/decode. Mesmo algoritmo usado em GPT-2/GPT-4.

### Fase 3 — Word2Vec
Embeddings de palavras via skip-gram com negative sampling. Treinamento em corpus português, visualização por similaridade cosseno.

### Fase 4 — Feed-Forward Language Model (FFLM)
Primeira rede neural de linguagem: concatena embeddings de N tokens de contexto, passa por camadas densas, prevê próximo token via cross-entropy.

### Fase 5 — Transformer (decoder-only)
Arquitetura GPT-style completa:
- Multi-Head Causal Self-Attention (Q/K/V projetados, máscara triangular)
- Feed-Forward com ReLU (FFN_HID=512)
- LayerNorm + conexões residuais
- Embeddings posicionais aprendidos
- Backpropagation completo através da atenção
- Otimizador Adam com bias correction
- Geração com temperatura, top-p nucleus sampling e repetition penalty

Configuração padrão: EMBED=256, HEADS=8, FFN=512, LAYERS=6, CTX=64.

### Fase 6 — Quantização INT8/INT4
Compressão de pesos para inferência eficiente:
- **Q8_0**: simétrico por linha, escala = max(|linha|)/127, 1 byte/peso
- **Q4_0**: blocos de 32, escala por bloco, nibbles com offset +8, 0.5 byte/peso
- `q8_matvec`: multiplicação matriz-vetor quantizada com acumulador int32
- Métricas: SNR, erro máximo, similaridade cosseno

### Pipeline — Chat Interativo
Integra todas as fases: treina em corpus de chat em português, salva/carrega `model.bin`, chat com formato `user <msg> assistant <resposta>`, parada automática no token `user`.

## Compilação

Cada fase tem seu próprio Makefile:

```bash
cd phase5 && make
cd phase6 && make
cd pipeline && make
```

## Pipeline (chat)

```bash
# Gerar corpus de treinamento
python generate_chat_corpus.py

# Compilar e executar
cd pipeline && make && ./pipeline.exe
```

Comandos disponíveis no chat:
```
t=0.8      — temperatura
p=0.9      — top-p nucleus sampling
r=1.3      — repetition penalty
n=20       — tokens a gerar
retreinar  — apaga model.bin e retreina
sair       — encerrar
```

## Pré-requisitos

- GCC (MinGW no Windows ou gcc no Linux/macOS)
- Python 3 (somente para gerar o corpus)
- Corpus de texto em português (`pt_BR.txt`) ou usar o gerador incluído

## Referências

- Vaswani et al. (2017) — *Attention Is All You Need*
- Mikolov et al. (2013) — *Efficient Estimation of Word Representations in Vector Space*
- Sennrich et al. (2016) — *Neural Machine Translation of Rare Words with Subword Units*
- Frantar et al. (2022) — *GPTQ: Accurate Post-Training Quantization*
