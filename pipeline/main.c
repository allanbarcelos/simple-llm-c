/*
 * Pipeline — LLM chat interativo
 *
 * Uso:  ./pipeline.exe [corpus] [epochs] [lr]
 *       Padrão: ../chat_corpus.txt  20  0.001
 *
 * Gerar corpus antes:
 *   python ../generate_chat_corpus.py
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../phase5/transformer.h"
#include "../phase6/quant.h"

#define MAX_FILE  (30 * 1024 * 1024)
#define MODEL_BIN "model.bin"

/* ── I/O ─────────────────────────────────────────────────────────────────── */
static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    if (sz > MAX_FILE) sz = MAX_FILE;
    char *buf = malloc(sz+1);
    buf[fread(buf,1,sz,f)] = '\0';
    fclose(f); return buf;
}
static void strip_nl(char *s) {
    int n=(int)strlen(s);
    while(n>0&&(s[n-1]=='\n'||s[n-1]=='\r'||s[n-1]==' ')) s[--n]='\0';
}

/* ── Salvar / Carregar modelo ────────────────────────────────────────────── */
static int model_save(const TransformerLM *m, const char *path) {
    FILE *f = fopen(path,"wb"); if(!f) return 0;
    fwrite(&m->vocab_size,sizeof(int),1,f);
    fwrite(m->emb,    sizeof(float),(size_t)m->vocab_size*TF_EMBED,f);
    fwrite(m->pos_emb,sizeof(float),TF_CONTEXT*TF_EMBED,f);
    fwrite(m->lm_head,sizeof(float),(size_t)m->vocab_size*TF_EMBED,f);
    fwrite(m->ln_f_g, sizeof(float),TF_EMBED,f);
    fwrite(m->ln_f_b, sizeof(float),TF_EMBED,f);
    for(int l=0;l<TF_LAYERS;l++) {
        const TFBlock *b=&m->blocks[l];
        int ee=TF_EMBED*TF_EMBED, ef=TF_FFN_HID*TF_EMBED;
        fwrite(b->WQ,sizeof(float),ee,f); fwrite(b->WK,sizeof(float),ee,f);
        fwrite(b->WV,sizeof(float),ee,f); fwrite(b->WO,sizeof(float),ee,f);
        fwrite(b->bQ,sizeof(float),TF_EMBED,f); fwrite(b->bK,sizeof(float),TF_EMBED,f);
        fwrite(b->bV,sizeof(float),TF_EMBED,f); fwrite(b->bO,sizeof(float),TF_EMBED,f);
        fwrite(b->W1,sizeof(float),ef,f); fwrite(b->b1,sizeof(float),TF_FFN_HID,f);
        fwrite(b->W2,sizeof(float),ef,f); fwrite(b->b2,sizeof(float),TF_EMBED,f);
        fwrite(b->ln1_g,sizeof(float),TF_EMBED,f); fwrite(b->ln1_b,sizeof(float),TF_EMBED,f);
        fwrite(b->ln2_g,sizeof(float),TF_EMBED,f); fwrite(b->ln2_b,sizeof(float),TF_EMBED,f);
    }
    fclose(f); return 1;
}
static int model_load(TransformerLM *m, const char *path) {
    FILE *f = fopen(path,"rb"); if(!f) return 0;
    int vs; fread(&vs,sizeof(int),1,f);
    if(vs!=m->vocab_size){fclose(f);return 0;}
    fread(m->emb,    sizeof(float),(size_t)vs*TF_EMBED,f);
    fread(m->pos_emb,sizeof(float),TF_CONTEXT*TF_EMBED,f);
    fread(m->lm_head,sizeof(float),(size_t)vs*TF_EMBED,f);
    fread(m->ln_f_g, sizeof(float),TF_EMBED,f);
    fread(m->ln_f_b, sizeof(float),TF_EMBED,f);
    for(int l=0;l<TF_LAYERS;l++) {
        TFBlock *b=&m->blocks[l];
        int ee=TF_EMBED*TF_EMBED, ef=TF_FFN_HID*TF_EMBED;
        fread(b->WQ,sizeof(float),ee,f); fread(b->WK,sizeof(float),ee,f);
        fread(b->WV,sizeof(float),ee,f); fread(b->WO,sizeof(float),ee,f);
        fread(b->bQ,sizeof(float),TF_EMBED,f); fread(b->bK,sizeof(float),TF_EMBED,f);
        fread(b->bV,sizeof(float),TF_EMBED,f); fread(b->bO,sizeof(float),TF_EMBED,f);
        fread(b->W1,sizeof(float),ef,f); fread(b->b1,sizeof(float),TF_FFN_HID,f);
        fread(b->W2,sizeof(float),ef,f); fread(b->b2,sizeof(float),TF_EMBED,f);
        fread(b->ln1_g,sizeof(float),TF_EMBED,f); fread(b->ln1_b,sizeof(float),TF_EMBED,f);
        fread(b->ln2_g,sizeof(float),TF_EMBED,f); fread(b->ln2_b,sizeof(float),TF_EMBED,f);
    }
    fclose(f); return 1;
}

/* ── main ────────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    /* Tentar chat_corpus.txt como padrão; fallback para pt_BR.txt */
    const char *default_corpus = "../chat_corpus.txt";
    FILE *fc = fopen(default_corpus,"rb");
    if(!fc) default_corpus = "../pt_BR.txt";
    else fclose(fc);

    const char *corpus_path = argc>1 ? argv[1] : default_corpus;
    int   epochs = argc>2 ? atoi(argv[2]) : 20;
    float lr     = argc>3 ? atof(argv[3]) : 0.001f;

    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  LLM Chat — Transformer + Quantização            ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");

    /* ── Corpus ── */
    printf("[1/4] Carregando corpus: %s\n", corpus_path);
    char *text = read_file(corpus_path);
    if(!text){fprintf(stderr,"Erro: nao foi possivel abrir '%s'\n",corpus_path);return 1;}
    printf("      %.1f KB carregados\n\n", (float)strlen(text)/1024);

    /* ── Vocabulário ── */
    printf("[2/4] Construindo vocabulário...\n");
    TFVocab *vocab = tfv_new();
    tfv_build(vocab, text, 1);
    printf("      %d palavras  (inclui tokens 'user' e 'assistant')\n", vocab->size);

    /* IDs dos tokens especiais */
    int user_id      = tfv_id(vocab, "user");
    int assistant_id = tfv_id(vocab, "assistant");
    printf("      token 'user'=%d  'assistant'=%d\n\n", user_id, assistant_id);

    int n_tok;
    int *tokens = tfv_tokenize(vocab, text, &n_tok);
    printf("      %d tokens para treino\n\n", n_tok);

    /* ── Modelo ── */
    int V = vocab->size < TF_MAX_VOCAB ? vocab->size : TF_MAX_VOCAB;
    TransformerLM *model = tf_new(V);
    printf("      Arquitetura: EMBED=%d  HEADS=%d  FFN=%d  LAYERS=%d  CTX=%d\n",
           TF_EMBED, TF_HEADS, TF_FFN_HID, TF_LAYERS, TF_CONTEXT);
    printf("      Parâmetros: %ld\n\n", tf_param_count(model));

    /* ── Treino ou carrega ── */
    if(model_load(model, MODEL_BIN)) {
        printf("[3/4] Modelo carregado de '%s'\n\n", MODEL_BIN);
    } else {
        printf("[3/4] Treinando (%d épocas, lr=%.4f)...\n\n", epochs, lr);
        tf_train(model, tokens, n_tok, epochs, lr, 1);
        if(model_save(model, MODEL_BIN))
            printf("\n      Modelo salvo em '%s'\n", MODEL_BIN);
        printf("\n");
    }

    /* ── Quantização ── */
    printf("[4/4] Quantização dos pesos:\n");
    long f32p = tf_param_count(model);
    const TFBlock *b0 = &model->blocks[0];
    int ef = TF_FFN_HID*TF_EMBED;
    Q8Matrix *w1q8 = q8_quantize(b0->W1, TF_FFN_HID, TF_EMBED);
    Q4Matrix *w1q4 = q4_quantize(b0->W1, TF_FFN_HID, TF_EMBED);
    float *dq8 = q8_dequantize(w1q8), *dq4 = q4_dequantize(w1q4);
    printf("      Parâmetros: %ld  |  F32: %ldKB  |  Q8≈%ldKB  |  Q4≈%ldKB\n",
           f32p, f32p*4/1024,
           (f32p/4*1)/1024,   /* ~1 byte/param para pesos principais */
           (f32p/4*1)/2/1024);
    printf("      Qualidade W1 bloco-0:  Q8 cos=%.6f  Q4 cos=%.6f\n\n",
           quant_cos_sim(b0->W1,dq8,ef), quant_cos_sim(b0->W1,dq4,ef));
    q8_free(w1q8); q4_free(w1q4); free(dq8); free(dq4);

    /* ── Chat ── */
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  Chat Interativo                                 ║\n");
    printf("║  Digite sua mensagem e pressione Enter.          ║\n");
    printf("║  Comandos:                                       ║\n");
    printf("║    sair          — encerrar                      ║\n");
    printf("║    t=0.8         — temperatura (padrão: 1.0)     ║\n");
    printf("║    p=0.9         — top-p nucleus (padrão: 0.9)   ║\n");
    printf("║    r=1.3         — repetition penalty (padrão)   ║\n");
    printf("║    n=20          — tokens a gerar (padrão: 20)   ║\n");
    printf("║    retreinar     — apaga model.bin e retreina    ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");

    /* configurações padrão */
    TFGenConfig cfg = { 1.0f, 0.9f, 1.3f, user_id };
    int gen_len = 20;
    char input[512];
    char prompt[640];

    while(1) {
        printf("Você: "); fflush(stdout);
        if(!fgets(input, sizeof(input), stdin)) break;
        strip_nl(input);
        if(!input[0]) continue;

        /* comandos de configuração */
        if(!strcmp(input,"sair")||!strcmp(input,"quit")) break;
        if(!strncmp(input,"t=",2)){cfg.temperature=(float)atof(input+2);printf("  temperatura=%.2f\n\n",cfg.temperature);continue;}
        if(!strncmp(input,"p=",2)){cfg.top_p=(float)atof(input+2);printf("  top-p=%.2f\n\n",cfg.top_p);continue;}
        if(!strncmp(input,"r=",2)){cfg.rep_penalty=(float)atof(input+2);printf("  rep_penalty=%.2f\n\n",cfg.rep_penalty);continue;}
        if(!strncmp(input,"n=",2)){gen_len=atoi(input+2);if(gen_len<1)gen_len=1;if(gen_len>100)gen_len=100;printf("  gen_len=%d\n\n",gen_len);continue;}
        if(!strcmp(input,"retreinar")){remove(MODEL_BIN);printf("  model.bin removido. Reinicie para retreinar.\n\n");continue;}

        /* construir prompt no formato de treino: "user <input> assistant" */
        if(assistant_id>=0)
            snprintf(prompt, sizeof(prompt), "user %s assistant", input);
        else
            snprintf(prompt, sizeof(prompt), "%s", input);

        int ns;
        int *seed = tfv_tokenize(vocab, prompt, &ns);
        if(ns==0){
            printf("  (nenhuma palavra reconhecida no vocabulário)\n\n");
            free(seed); continue;
        }

        printf("LLM:  ");
        tf_generate(model, vocab, seed, ns, gen_len, &cfg);
        printf("\n");
        free(seed);
    }

    printf("\nEncerrando.\n");
    tf_free(model); tfv_free(vocab);
    free(tokens); free(text);
    return 0;
}
