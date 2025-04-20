/*
 * Fase 5 — Transformer Language Model (decoder-only, GPT-style)
 *
 * Uso:  ./phase5_demo [arquivo] [epochs] [lr]
 *       Padrão: ../pt_BR.txt  5  0.001
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "transformer.h"

#define MAX_FILE (20 * 1024 * 1024)

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f); rewind(f);
    if (sz > MAX_FILE) sz = MAX_FILE;
    char *buf = malloc(sz+1);
    buf[fread(buf,1,sz,f)] = '\0';
    fclose(f);
    return buf;
}

static void section(const char *t) {
    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║  %-48s║\n", t);
    printf("╚══════════════════════════════════════════════════╝\n\n");
}

int main(int argc, char *argv[]) {
    const char *path   = argc > 1 ? argv[1] : "../pt_BR.txt";
    int         epochs = argc > 2 ? atoi(argv[2]) : 5;
    float       lr     = argc > 3 ? atof(argv[3]) : 0.001f;

    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  FASE 5 — Transformer Language Model (GPT-style) ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");

    char *text = read_file(path);
    if (!text) { fprintf(stderr,"Erro ao abrir '%s'\n", path); return 1; }
    printf("\nCorpus: %s\n", path);

    /* ── O que é Self-Attention? ─────────────────────────────────────── */
    section("O problema que a atenção resolve");
    printf("Fase 4 (FFLM): contexto fixo de %d tokens concatenados.\n", 3);
    printf("  'quem ferro FERE...' — 'fere' não 'vê' 'quem'!\n\n");
    printf("Transformer: cada token presta atenção em TODOS os anteriores.\n");
    printf("  'fere' pode olhar diretamente para 'quem' e 'ferro'.\n\n");
    printf("Mecanismo Query-Key-Value:\n");
    printf("  Q = 'o que eu estou procurando?'  (token atual)\n");
    printf("  K = 'o que eu tenho para oferecer?' (todos tokens)\n");
    printf("  V = 'qual informação eu compartilho?' (todos tokens)\n\n");
    printf("  score(t,s) = Q[t] · K[s] / √d_head\n");
    printf("  attn(t,s)  = softmax(score)          (peso de atenção)\n");
    printf("  out[t]     = Σ_s attn(t,s) * V[s]   (contexto ponderado)\n\n");

    /* ── Máscara Causal ──────────────────────────────────────────────── */
    section("Máscara causal (decoder-only)");
    printf("Geração autoregressiva: token t só pode ver t-1, t-2, ...\n");
    printf("Scores para posições futuras → -∞ antes do softmax:\n\n");
    printf("  Pos: 0    1    2    3\n");
    printf("  0: [vis, -∞,  -∞,  -∞ ]\n");
    printf("  1: [vis, vis, -∞,  -∞ ]\n");
    printf("  2: [vis, vis, vis, -∞ ]\n");
    printf("  3: [vis, vis, vis, vis]\n\n");
    printf("Após softmax de -∞: probabilidade → 0 (token invisível).\n\n");

    /* ── Multi-Head Attention ────────────────────────────────────────── */
    section("Multi-Head Attention");
    printf("Em vez de uma atenção com d=%d dimensões,\n", TF_EMBED);
    printf("usamos %d cabeças de d=%d dimensões em paralelo.\n\n",
           TF_HEADS, TF_HEAD_DIM);
    printf("Cada cabeça aprende um tipo diferente de relação:\n");
    printf("  Cabeça 1: relações sintáticas (sujeito → verbo)\n");
    printf("  Cabeça 2: relações semânticas (palavra → sinônimo)\n");
    printf("  Cabeça 3: relações posicionais (próximo vs distante)\n");
    printf("  Cabeça 4: outras relações emergentes...\n\n");
    printf("Saídas das cabeças são concatenadas e projetadas com WO.\n\n");

    /* ── Arquitetura completa ────────────────────────────────────────── */
    section("Arquitetura do modelo");
    printf("Configuração deste modelo:\n");
    printf("  EMBED_DIM  = %d\n", TF_EMBED);
    printf("  HEADS      = %d  (head_dim = %d)\n", TF_HEADS, TF_HEAD_DIM);
    printf("  FFN_HIDDEN = %d\n", TF_FFN_HID);
    printf("  LAYERS     = %d\n", TF_LAYERS);
    printf("  CONTEXT    = %d tokens\n\n", TF_CONTEXT);
    printf("Por bloco Transformer:\n");
    printf("  x → LayerNorm → Q,K,V proj → Masked Attn → WO → +x (residual)\n");
    printf("    → LayerNorm → W1·ReLU·W2 → +x (residual)\n\n");
    printf("  LayerNorm: normaliza média/variância por posição\n");
    printf("             garante gradientes estáveis em profundidade\n\n");
    printf("  Residual:  x = x + sublayer(x)\n");
    printf("             gradiente flui direto (sem desaparecer)\n\n");

    /* ── Vocabulário e tokenização ───────────────────────────────────── */
    section("Vocabulário");
    TFVocab *vocab = tfv_new();
    tfv_build(vocab, text, 1);
    printf("Vocabulário: %d palavras únicas\n", vocab->size);
    if (vocab->size > 20) {
        printf("Primeiras 20: ");
        for (int i=0;i<20&&i<vocab->size;i++) printf("%s ", vocab->by_id[i]->word);
        printf("\n");
    }

    /* tokenizar */
    int n_tokens;
    int *tokens = tfv_tokenize(vocab, text, &n_tokens);
    printf("Tokens: %d\n\n", n_tokens);

    /* ── Criar modelo ────────────────────────────────────────────────── */
    int V = vocab->size < TF_MAX_VOCAB ? vocab->size : TF_MAX_VOCAB;
    TransformerLM *model = tf_new(V);
    printf("Parâmetros totais: %ld\n\n", tf_param_count(model));

    /* ── Treinamento ─────────────────────────────────────────────────── */
    section("Treinamento");
    printf("Objetivo: prever próximo token dada janela de %d tokens.\n", TF_CONTEXT);
    printf("Loss: cross-entropy  |  Métrica: perplexidade = e^loss\n\n");
    printf("Perplexidade = quantas palavras plausíveis o modelo considera.\n");
    printf("  PPL=1000 → modelo praticamente aleatório\n");
    printf("  PPL=100  → começando a aprender\n");
    printf("  PPL=10   → bom para corpus pequeno\n\n");
    printf("Treinando %d época(s), lr=%.4f ...\n\n", epochs, lr);
    tf_train(model, tokens, n_tokens, epochs, lr, 1);

    /* ── Geração de texto ────────────────────────────────────────────── */
    section("Geração de texto");
    printf("Temperatura controla aleatoriedade:\n");
    printf("  T→0: sempre escolhe token mais provável (repetitivo)\n");
    printf("  T=1: distribuição original do modelo\n");
    printf("  T>1: mais aleatório, criativo mas menos coerente\n\n");

    struct { const char *prompt; float temp; float top_p; } tests[] = {
        { "quem ferro", 0.7f, 1.0f },
        { "mais vale",  1.0f, 0.9f },
        { "o amor",     1.2f, 0.9f },
        { NULL, 0, 0 }
    };

    for (int i=0; tests[i].prompt; i++) {
        int ns; int *seed = tfv_tokenize(vocab, tests[i].prompt, &ns);
        if (ns == 0) { free(seed); continue; }
        printf("  Prompt: \"%s\"  (T=%.1f)\n  → ", tests[i].prompt, tests[i].temp);
        TFGenConfig cfg = { tests[i].temp, tests[i].top_p, 1.3f, -1 };
        tf_generate(model, vocab, seed, ns, 12, &cfg);
        free(seed);
    }

    /* ── Comparação com fases anteriores ───────────────────────────── */
    section("Evolução ao longo das fases");
    printf("  Fase 1 — Matemática: linear algebra, softmax, backprop\n");
    printf("  Fase 2 — Tokenizador: BPE, subpalavras\n");
    printf("  Fase 3 — Word2Vec: embeddings densos, semântica\n");
    printf("  Fase 4 — FFLM: primeira rede neural de linguagem\n");
    printf("  Fase 5 — Transformer: atenção, contexto longo ← AGORA\n\n");
    printf("  Diferença chave Fase 4 → 5:\n");
    printf("    FFLM:        contexto fixo concatenado (3 tokens × 32 dim)\n");
    printf("    Transformer: QUALQUER posição pode atender qualquer outra\n\n");

    /* ── Próximos passos ─────────────────────────────────────────────── */
    section("Rumo a um LLM de produção");
    printf("O que falta para escalar:\n\n");
    printf("  Fase 6 — Quantização (INT8/INT4):\n");
    printf("    float32 → int8: 4× menos memória, 2-4× mais rápido na CPU\n\n");
    printf("  Escala:\n");
    printf("    GPT-2 small: 12 layers, 12 heads, 768 dim → 117M parâmetros\n");
    printf("    Llama 3.2 1B: CPU-executável com quantização\n\n");
    printf("  Melhorias de arquitetura:\n");
    printf("    RoPE (positional encoding rotacional)\n");
    printf("    GQA (grouped-query attention, mais eficiente)\n");
    printf("    SwiGLU (ativação no FFN)\n");
    printf("    RMSNorm (mais simples que LayerNorm)\n\n");

    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  Fase 5 concluída — Transformer implementado!   ║\n");
    printf("║  Próximo: Fase 6 — Quantização INT8/INT4        ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");

    tf_free(model);
    tfv_free(vocab);
    free(tokens);
    free(text);
    return 0;
}
