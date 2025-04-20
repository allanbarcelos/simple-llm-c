#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <time.h>
#include "transformer.h"

/* ── utilidades ─────────────────────────────────────────────────────────── */
static float randf(void) { return (float)rand() / RAND_MAX; }

static void xavier(float *w, int n, int fan_in) {
    float s = sqrtf(2.0f / (float)fan_in);
    for (int i = 0; i < n; i++) w[i] = (randf()*2.0f - 1.0f) * s;
}

static TFAdamSlot adam_new(int n) {
    TFAdamSlot a; a.n = n;
    a.m = calloc((size_t)n, sizeof(float));
    a.v = calloc((size_t)n, sizeof(float));
    return a;
}
static void adam_free(TFAdamSlot *a) { free(a->m); free(a->v); }

static void adam_step(TFAdamSlot *a, float *p, const float *g,
                      int n, float lr, int t) {
    const float b1=0.9f, b2=0.999f, eps=1e-8f;
    float bc1 = 1.0f - powf(b1, (float)t);
    float bc2 = 1.0f - powf(b2, (float)t);
    for (int i = 0; i < n; i++) {
        a->m[i] = b1*a->m[i] + (1.0f-b1)*g[i];
        a->v[i] = b2*a->v[i] + (1.0f-b2)*g[i]*g[i];
        p[i] -= lr * (a->m[i]/bc1) / (sqrtf(a->v[i]/bc2) + eps);
    }
}

/* ── vocabulário ─────────────────────────────────────────────────────────── */
static unsigned long djb2(const char *s) {
    unsigned long h = 5381;
    while (*s) h = ((h<<5)+h) + (unsigned char)*s++;
    return h;
}

static void norm_word(char *dst, const char *src) {
    char tmp[TF_VOCAB_WORD]; int tl=0, i=0;
    while (src[i] && ispunct((unsigned char)src[i])) i++;
    while (src[i] && !isspace((unsigned char)src[i]) && tl < TF_VOCAB_WORD-1) {
        unsigned char c = (unsigned char)src[i++];
        tmp[tl++] = (c < 128) ? (char)tolower(c) : (char)c;
    }
    while (tl > 0 && ispunct((unsigned char)tmp[tl-1])) tl--;
    tmp[tl] = '\0';
    strncpy(dst, tmp, TF_VOCAB_WORD-1);
    dst[TF_VOCAB_WORD-1] = '\0';
}

TFVocab *tfv_new(void) {
    TFVocab *v = malloc(sizeof(TFVocab));
    v->hash  = calloc(TF_VOCAB_HASH, sizeof(TFVEntry*));
    v->by_id = NULL; v->size = 0;
    return v;
}
void tfv_free(TFVocab *v) {
    for (int i=0;i<TF_VOCAB_HASH;i++) {
        TFVEntry *e=v->hash[i]; while(e){TFVEntry*n=e->next;free(e);e=n;}
    }
    free(v->hash); free(v->by_id); free(v);
}
static TFVEntry *tfv_find(const TFVocab *v, const char *w) {
    unsigned long h = djb2(w) & (TF_VOCAB_HASH-1);
    for (TFVEntry *e=v->hash[h]; e; e=e->next)
        if (!strcmp(e->word,w)) return e;
    return NULL;
}
int tfv_id(const TFVocab *v, const char *w) {
    char buf[TF_VOCAB_WORD]; norm_word(buf,w);
    TFVEntry *e = tfv_find(v,buf);
    return e ? e->id : -1;
}
void tfv_build(TFVocab *v, const char *text, int min_freq) {
    char buf[TF_VOCAB_WORD], tmp[TF_VOCAB_WORD];
    const char *p=text;
    while (*p) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;
        int i=0;
        while (*p && !isspace((unsigned char)*p) && i<TF_VOCAB_WORD-1)
            tmp[i++] = *p++;
        tmp[i]='\0'; norm_word(buf,tmp);
        if (!buf[0]) continue;
        unsigned long h = djb2(buf)&(TF_VOCAB_HASH-1);
        TFVEntry *e = tfv_find(v,buf);
        if (!e) {
            e=calloc(1,sizeof(TFVEntry));
            strncpy(e->word,buf,TF_VOCAB_WORD-1);
            e->id=-1; e->next=v->hash[h]; v->hash[h]=e;
        }
        e->freq++;
    }
    int cnt=0;
    for (int i=0;i<TF_VOCAB_HASH&&cnt<TF_MAX_VOCAB;i++)
        for (TFVEntry *e=v->hash[i];e&&cnt<TF_MAX_VOCAB;e=e->next)
            if (e->freq>=min_freq) e->id=cnt++;
    v->size=cnt;
    v->by_id=malloc((size_t)cnt*sizeof(TFVEntry*));
    for (int i=0;i<TF_VOCAB_HASH;i++)
        for (TFVEntry *e=v->hash[i];e;e=e->next)
            if (e->id>=0) v->by_id[e->id]=e;
}
int *tfv_tokenize(const TFVocab *v, const char *text, int *out_n) {
    int cap=1024,n=0;
    int *ids=malloc((size_t)cap*sizeof(int));
    char tmp[TF_VOCAB_WORD],buf[TF_VOCAB_WORD];
    const char *p=text;
    while (*p) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;
        int i=0;
        while (*p && !isspace((unsigned char)*p) && i<TF_VOCAB_WORD-1)
            tmp[i++]=*p++;
        tmp[i]='\0'; norm_word(buf,tmp);
        if (!buf[0]) continue;
        int id=tfv_id(v,buf);
        if (id<0) continue;
        if (n==cap){cap*=2;ids=realloc(ids,(size_t)cap*sizeof(int));}
        ids[n++]=id;
    }
    *out_n=n; return ids;
}

/* ── LayerNorm ───────────────────────────────────────────────────────────── */
static void layernorm(const float *x, const float *g, const float *b,
                      float *y, float *mo, float *ro, int d) {
    float mean=0,var=0;
    for (int i=0;i<d;i++) mean+=x[i];
    mean/=(float)d;
    for (int i=0;i<d;i++){float df=x[i]-mean;var+=df*df;}
    float rstd=1.0f/sqrtf(var/(float)d+1e-5f);
    if (mo) *mo=mean; if (ro) *ro=rstd;
    for (int i=0;i<d;i++) y[i]=(x[i]-mean)*rstd*g[i]+b[i];
}
static void layernorm_bwd(const float *dy, const float *x, const float *g,
                          float mean, float rstd,
                          float *dx, float *dg, float *db, int d) {
    float xhat[TF_EMBED], dot=0, sum=0;
    for (int i=0;i<d;i++){
        xhat[i]=(x[i]-mean)*rstd;
        dot+=dy[i]*g[i]*xhat[i]; sum+=dy[i]*g[i];
    }
    for (int i=0;i<d;i++){
        dx[i]+=rstd*(dy[i]*g[i]-(sum+xhat[i]*dot)/(float)d);
        dg[i]+=dy[i]*xhat[i]; db[i]+=dy[i];
    }
}
static void softmax_row(float *x, int n) {
    float mx=x[0]; for(int i=1;i<n;i++) if(x[i]>mx) mx=x[i];
    float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-mx);s+=x[i];}
    for(int i=0;i<n;i++) x[i]/=s;
}

/* ── Bloco: init / free ──────────────────────────────────────────────────── */
static void block_init(TFBlock *b) {
    int ee=TF_EMBED*TF_EMBED, ef=TF_FFN_HID*TF_EMBED;
    b->WQ=malloc((size_t)ee*4); xavier(b->WQ,ee,TF_EMBED);
    b->WK=malloc((size_t)ee*4); xavier(b->WK,ee,TF_EMBED);
    b->WV=malloc((size_t)ee*4); xavier(b->WV,ee,TF_EMBED);
    b->WO=malloc((size_t)ee*4); xavier(b->WO,ee,TF_EMBED);
    b->bQ=calloc(TF_EMBED,4); b->bK=calloc(TF_EMBED,4);
    b->bV=calloc(TF_EMBED,4); b->bO=calloc(TF_EMBED,4);
    b->W1=malloc((size_t)ef*4); xavier(b->W1,ef,TF_EMBED);
    b->b1=calloc(TF_FFN_HID,4);
    b->W2=malloc((size_t)ef*4); xavier(b->W2,ef,TF_FFN_HID);
    b->b2=calloc(TF_EMBED,4);
    b->ln1_g=malloc(TF_EMBED*4); b->ln1_b=calloc(TF_EMBED,4);
    b->ln2_g=malloc(TF_EMBED*4); b->ln2_b=calloc(TF_EMBED,4);
    for(int i=0;i<TF_EMBED;i++) b->ln1_g[i]=b->ln2_g[i]=1.0f;
    b->aWQ=adam_new(ee); b->aWK=adam_new(ee);
    b->aWV=adam_new(ee); b->aWO=adam_new(ee);
    b->abQ=adam_new(TF_EMBED); b->abK=adam_new(TF_EMBED);
    b->abV=adam_new(TF_EMBED); b->abO=adam_new(TF_EMBED);
    b->aW1=adam_new(ef); b->ab1=adam_new(TF_FFN_HID);
    b->aW2=adam_new(ef); b->ab2=adam_new(TF_EMBED);
    b->aln1g=adam_new(TF_EMBED); b->aln1b=adam_new(TF_EMBED);
    b->aln2g=adam_new(TF_EMBED); b->aln2b=adam_new(TF_EMBED);
}
static void block_free(TFBlock *b) {
    free(b->WQ);free(b->WK);free(b->WV);free(b->WO);
    free(b->bQ);free(b->bK);free(b->bV);free(b->bO);
    free(b->W1);free(b->b1);free(b->W2);free(b->b2);
    free(b->ln1_g);free(b->ln1_b);free(b->ln2_g);free(b->ln2_b);
    adam_free(&b->aWQ);adam_free(&b->aWK);adam_free(&b->aWV);adam_free(&b->aWO);
    adam_free(&b->abQ);adam_free(&b->abK);adam_free(&b->abV);adam_free(&b->abO);
    adam_free(&b->aW1);adam_free(&b->ab1);adam_free(&b->aW2);adam_free(&b->ab2);
    adam_free(&b->aln1g);adam_free(&b->aln1b);
    adam_free(&b->aln2g);adam_free(&b->aln2b);
}

/* ── Modelo: criar / libertar ────────────────────────────────────────────── */
TransformerLM *tf_new(int vocab_size) {
    srand((unsigned)time(NULL));
    TransformerLM *m = calloc(1,sizeof(TransformerLM));
    m->vocab_size=vocab_size;
    int ve=vocab_size*TF_EMBED, ce=TF_CONTEXT*TF_EMBED;
    m->emb    =malloc((size_t)ve*4); xavier(m->emb,    ve,TF_EMBED);
    m->pos_emb=malloc((size_t)ce*4); xavier(m->pos_emb,ce,TF_CONTEXT);
    m->lm_head=malloc((size_t)ve*4); xavier(m->lm_head,ve,TF_EMBED);
    m->ln_f_g =malloc(TF_EMBED*4);   m->ln_f_b=calloc(TF_EMBED,4);
    for(int i=0;i<TF_EMBED;i++) m->ln_f_g[i]=1.0f;
    for(int l=0;l<TF_LAYERS;l++) block_init(&m->blocks[l]);
    m->a_emb=adam_new(ve); m->a_pos=adam_new(ce);
    m->a_lmh=adam_new(ve); m->a_lfg=adam_new(TF_EMBED); m->a_lfb=adam_new(TF_EMBED);
    m->adam_t=0;
    return m;
}
void tf_free(TransformerLM *m) {
    free(m->emb);free(m->pos_emb);free(m->lm_head);free(m->ln_f_g);free(m->ln_f_b);
    for(int l=0;l<TF_LAYERS;l++) block_free(&m->blocks[l]);
    adam_free(&m->a_emb);adam_free(&m->a_pos);adam_free(&m->a_lmh);
    adam_free(&m->a_lfg);adam_free(&m->a_lfb);
    free(m);
}
long tf_param_count(const TransformerLM *m) {
    long n=(long)m->vocab_size*TF_EMBED*2+TF_CONTEXT*TF_EMBED+2*TF_EMBED;
    n+=(long)TF_LAYERS*(4*TF_EMBED*TF_EMBED+4*TF_EMBED
        +2*TF_FFN_HID*TF_EMBED+TF_FFN_HID+TF_EMBED+4*TF_EMBED);
    return n;
}

/* ── Forward de um bloco ─────────────────────────────────────────────────── */
static void block_forward(const TFBlock *b, int seq,
                           const float in[][TF_EMBED],
                           float out[][TF_EMBED], TFBlockCache *c) {
    memcpy(c->x, in, (size_t)seq*TF_EMBED*sizeof(float));
    for(int t=0;t<seq;t++)
        layernorm(c->x[t],b->ln1_g,b->ln1_b,c->xn1[t],NULL,NULL,TF_EMBED);
    for(int t=0;t<seq;t++)
        for(int j=0;j<TF_EMBED;j++) {
            float q=b->bQ[j],k=b->bK[j],v=b->bV[j];
            for(int i=0;i<TF_EMBED;i++){
                q+=c->xn1[t][i]*b->WQ[i*TF_EMBED+j];
                k+=c->xn1[t][i]*b->WK[i*TF_EMBED+j];
                v+=c->xn1[t][i]*b->WV[i*TF_EMBED+j];
            }
            c->Q[t][j]=q; c->K[t][j]=k; c->V[t][j]=v;
        }
    float scale=1.0f/sqrtf((float)TF_HEAD_DIM);
    memset(c->av,0,(size_t)seq*TF_EMBED*sizeof(float));
    for(int h=0;h<TF_HEADS;h++) {
        int off=h*TF_HEAD_DIM;
        for(int t=0;t<seq;t++) {
            for(int s=0;s<seq;s++) {
                float sc=0;
                for(int d=0;d<TF_HEAD_DIM;d++) sc+=c->Q[t][off+d]*c->K[s][off+d];
                c->scores[h][t][s]=(s<=t)?sc*scale:-1e9f;
            }
            memcpy(c->attn[h][t],c->scores[h][t],(size_t)seq*sizeof(float));
            softmax_row(c->attn[h][t],seq);
        }
        for(int t=0;t<seq;t++)
            for(int s=0;s<seq;s++)
                for(int d=0;d<TF_HEAD_DIM;d++)
                    c->av[t][off+d]+=c->attn[h][t][s]*c->V[s][off+d];
    }
    for(int t=0;t<seq;t++)
        for(int j=0;j<TF_EMBED;j++) {
            float val=b->bO[j];
            for(int i=0;i<TF_EMBED;i++) val+=c->av[t][i]*b->WO[i*TF_EMBED+j];
            c->x2[t][j]=c->x[t][j]+val;
        }
    for(int t=0;t<seq;t++)
        layernorm(c->x2[t],b->ln2_g,b->ln2_b,c->xn2[t],NULL,NULL,TF_EMBED);
    for(int t=0;t<seq;t++) {
        for(int j=0;j<TF_FFN_HID;j++) {
            float val=b->b1[j];
            for(int i=0;i<TF_EMBED;i++) val+=c->xn2[t][i]*b->W1[j*TF_EMBED+i];
            c->fz[t][j]=val; c->fh[t][j]=val>0?val:0.0f;
        }
        for(int j=0;j<TF_EMBED;j++) {
            float val=b->b2[j];
            for(int i=0;i<TF_FFN_HID;i++) val+=c->fh[t][i]*b->W2[j*TF_FFN_HID+i];
            out[t][j]=c->x2[t][j]+val;
        }
    }
}

/* ── Forward completo (buffers estáticos — sem heap por chamada) ─────────── */
static float s_layer[TF_LAYERS+1][TF_CONTEXT][TF_EMBED];

float *tf_forward(const TransformerLM *m, const int *tokens,
                  int seq, TFCache *cache) {
    for(int t=0;t<seq;t++)
        for(int d=0;d<TF_EMBED;d++)
            s_layer[0][t][d]=m->emb[tokens[t]*TF_EMBED+d]+m->pos_emb[t*TF_EMBED+d];
    memcpy(cache->emb_out,s_layer[0],(size_t)seq*TF_EMBED*sizeof(float));
    for(int l=0;l<TF_LAYERS;l++)
        block_forward(&m->blocks[l],seq,
                      (const float(*)[TF_EMBED])s_layer[l],
                      s_layer[l+1],&cache->bc[l]);
    for(int t=0;t<seq;t++)
        layernorm(s_layer[TF_LAYERS][t],m->ln_f_g,m->ln_f_b,
                  cache->final_h[t],NULL,NULL,TF_EMBED);
    int V=m->vocab_size;
    float *logits=malloc((size_t)seq*V*sizeof(float));
    for(int t=0;t<seq;t++)
        for(int v=0;v<V;v++) {
            float val=0;
            for(int d=0;d<TF_EMBED;d++) val+=cache->final_h[t][d]*m->lm_head[v*TF_EMBED+d];
            logits[t*V+v]=val;
        }
    return logits;
}

/* ── Backward de um bloco (todos os arrays seq×EMBED na heap) ────────────── */
#define IDX(t,d) ((t)*TF_EMBED+(d))
#define IDXF(t,d) ((t)*TF_FFN_HID+(d))

static void block_backward(TFBlock *b, int seq, TFBlockCache *c,
                            const float d_out[][TF_EMBED],
                            float d_in[][TF_EMBED],
                            float lr, int adam_t) {
    int ee=TF_EMBED*TF_EMBED, ef=TF_FFN_HID*TF_EMBED;

    /* Gradientes de pesos — heap */
    float *dWQ=calloc((size_t)ee,4); float *dWK=calloc((size_t)ee,4);
    float *dWV=calloc((size_t)ee,4); float *dWO=calloc((size_t)ee,4);
    float *dW1=calloc((size_t)ef,4); float *dW2=calloc((size_t)ef,4);

    /* Gradientes de ativações — heap (evita stack overflow com EMBED=128) */
    float *dx2  = calloc((size_t)seq*TF_EMBED,4);
    float *dxn2 = calloc((size_t)seq*TF_EMBED,4);
    float *dQ   = calloc((size_t)seq*TF_EMBED,4);
    float *dK   = calloc((size_t)seq*TF_EMBED,4);
    float *dV   = calloc((size_t)seq*TF_EMBED,4);
    float *dav  = calloc((size_t)seq*TF_EMBED,4);
    float *dxn1 = calloc((size_t)seq*TF_EMBED,4);

    float dbQ[TF_EMBED]={0},dbK[TF_EMBED]={0},dbV[TF_EMBED]={0},dbO[TF_EMBED]={0};
    float db1[TF_FFN_HID]={0},db2[TF_EMBED]={0};
    float dln1g[TF_EMBED]={0},dln1b[TF_EMBED]={0};
    float dln2g[TF_EMBED]={0},dln2b[TF_EMBED]={0};

    memset(d_in,0,(size_t)seq*TF_EMBED*sizeof(float));

    /* FFN backward */
    for(int t=0;t<seq;t++) {
        for(int j=0;j<TF_EMBED;j++) dx2[IDX(t,j)]+=d_out[t][j];
        float d_fh[TF_FFN_HID]={0};
        for(int i=0;i<TF_FFN_HID;i++)
            for(int j=0;j<TF_EMBED;j++) {
                dW2[j*TF_FFN_HID+i]+=c->fh[t][i]*d_out[t][j];
                d_fh[i]+=b->W2[j*TF_FFN_HID+i]*d_out[t][j];
            }
        for(int j=0;j<TF_EMBED;j++) db2[j]+=d_out[t][j];
        float d_fz[TF_FFN_HID];
        for(int i=0;i<TF_FFN_HID;i++) d_fz[i]=c->fz[t][i]>0?d_fh[i]:0.0f;
        for(int i=0;i<TF_FFN_HID;i++) {
            db1[i]+=d_fz[i];
            for(int j=0;j<TF_EMBED;j++) {
                dW1[i*TF_EMBED+j]+=c->xn2[t][j]*d_fz[i];
                dxn2[IDX(t,j)]+=b->W1[i*TF_EMBED+j]*d_fz[i];
            }
        }
    }

    /* LayerNorm2 backward */
    for(int t=0;t<seq;t++) {
        float mean=0,var=0;
        for(int j=0;j<TF_EMBED;j++) mean+=c->x2[t][j]; mean/=(float)TF_EMBED;
        for(int j=0;j<TF_EMBED;j++){float d=c->x2[t][j]-mean;var+=d*d;}
        float rstd=1.0f/sqrtf(var/(float)TF_EMBED+1e-5f);
        float tmp[TF_EMBED]={0};
        layernorm_bwd(&dxn2[IDX(t,0)],c->x2[t],b->ln2_g,mean,rstd,
                      tmp,dln2g,dln2b,TF_EMBED);
        for(int j=0;j<TF_EMBED;j++) dx2[IDX(t,j)]+=tmp[j];
    }

    /* Output proj backward */
    for(int t=0;t<seq;t++) {
        for(int j=0;j<TF_EMBED;j++) d_in[t][j]+=dx2[IDX(t,j)];
        for(int j=0;j<TF_EMBED;j++) dbO[j]+=dx2[IDX(t,j)];
        for(int i=0;i<TF_EMBED;i++)
            for(int j=0;j<TF_EMBED;j++) {
                dWO[i*TF_EMBED+j]+=c->av[t][i]*dx2[IDX(t,j)];
                dav[IDX(t,i)]+=b->WO[i*TF_EMBED+j]*dx2[IDX(t,j)];
            }
    }

    /* Multi-head attention backward */
    float sc=1.0f/sqrtf((float)TF_HEAD_DIM);
    float *dattn  = calloc((size_t)seq*seq,4);
    float *dscores= calloc((size_t)seq*seq,4);
    for(int h=0;h<TF_HEADS;h++) {
        int off=h*TF_HEAD_DIM;
        memset(dattn,0,(size_t)seq*seq*4);
        for(int t=0;t<seq;t++)
            for(int s=0;s<seq;s++)
                for(int d=0;d<TF_HEAD_DIM;d++) {
                    dattn[t*seq+s]+=dav[IDX(t,off+d)]*c->V[s][off+d];
                    dV[IDX(s,off+d)]+=c->attn[h][t][s]*dav[IDX(t,off+d)];
                }
        memset(dscores,0,(size_t)seq*seq*4);
        for(int t=0;t<seq;t++) {
            float dot=0;
            for(int s=0;s<=t;s++) dot+=dattn[t*seq+s]*c->attn[h][t][s];
            for(int s=0;s<=t;s++)
                dscores[t*seq+s]=c->attn[h][t][s]*(dattn[t*seq+s]-dot);
        }
        for(int t=0;t<seq;t++)
            for(int s=0;s<seq;s++) {
                float ds=dscores[t*seq+s]*sc;
                for(int d=0;d<TF_HEAD_DIM;d++) {
                    dQ[IDX(t,off+d)]+=ds*c->K[s][off+d];
                    dK[IDX(s,off+d)]+=ds*c->Q[t][off+d];
                }
            }
    }
    free(dattn); free(dscores);

    /* QKV proj backward */
    for(int t=0;t<seq;t++) {
        for(int j=0;j<TF_EMBED;j++){dbQ[j]+=dQ[IDX(t,j)];dbK[j]+=dK[IDX(t,j)];dbV[j]+=dV[IDX(t,j)];}
        for(int i=0;i<TF_EMBED;i++)
            for(int j=0;j<TF_EMBED;j++) {
                dWQ[i*TF_EMBED+j]+=c->xn1[t][i]*dQ[IDX(t,j)];
                dWK[i*TF_EMBED+j]+=c->xn1[t][i]*dK[IDX(t,j)];
                dWV[i*TF_EMBED+j]+=c->xn1[t][i]*dV[IDX(t,j)];
                dxn1[IDX(t,i)]+=b->WQ[i*TF_EMBED+j]*dQ[IDX(t,j)]
                               +b->WK[i*TF_EMBED+j]*dK[IDX(t,j)]
                               +b->WV[i*TF_EMBED+j]*dV[IDX(t,j)];
            }
    }

    /* LayerNorm1 backward */
    for(int t=0;t<seq;t++) {
        float mean=0,var=0;
        for(int j=0;j<TF_EMBED;j++) mean+=c->x[t][j]; mean/=(float)TF_EMBED;
        for(int j=0;j<TF_EMBED;j++){float d=c->x[t][j]-mean;var+=d*d;}
        float rstd=1.0f/sqrtf(var/(float)TF_EMBED+1e-5f);
        float tmp[TF_EMBED]={0};
        layernorm_bwd(&dxn1[IDX(t,0)],c->x[t],b->ln1_g,mean,rstd,
                      tmp,dln1g,dln1b,TF_EMBED);
        for(int j=0;j<TF_EMBED;j++) d_in[t][j]+=tmp[j];
    }

    adam_step(&b->aWQ,b->WQ,dWQ,ee,lr,adam_t);
    adam_step(&b->aWK,b->WK,dWK,ee,lr,adam_t);
    adam_step(&b->aWV,b->WV,dWV,ee,lr,adam_t);
    adam_step(&b->aWO,b->WO,dWO,ee,lr,adam_t);
    adam_step(&b->abQ,b->bQ,dbQ,TF_EMBED,lr,adam_t);
    adam_step(&b->abK,b->bK,dbK,TF_EMBED,lr,adam_t);
    adam_step(&b->abV,b->bV,dbV,TF_EMBED,lr,adam_t);
    adam_step(&b->abO,b->bO,dbO,TF_EMBED,lr,adam_t);
    adam_step(&b->aW1,b->W1,dW1,ef,lr,adam_t);
    adam_step(&b->ab1,b->b1,db1,TF_FFN_HID,lr,adam_t);
    adam_step(&b->aW2,b->W2,dW2,ef,lr,adam_t);
    adam_step(&b->ab2,b->b2,db2,TF_EMBED,lr,adam_t);
    adam_step(&b->aln1g,b->ln1_g,dln1g,TF_EMBED,lr,adam_t);
    adam_step(&b->aln1b,b->ln1_b,dln1b,TF_EMBED,lr,adam_t);
    adam_step(&b->aln2g,b->ln2_g,dln2g,TF_EMBED,lr,adam_t);
    adam_step(&b->aln2b,b->ln2_b,dln2b,TF_EMBED,lr,adam_t);

    free(dWQ);free(dWK);free(dWV);free(dWO);free(dW1);free(dW2);
    free(dx2);free(dxn2);free(dQ);free(dK);free(dV);free(dav);free(dxn1);
}

/* ── Backward completo ───────────────────────────────────────────────────── */
float tf_backward(TransformerLM *m, TFCache *cache,
                  const float *logits, const int *tokens,
                  int seq, float lr) {
    m->adam_t++;
    int V=m->vocab_size; float loss=0;
    float *d_logits=calloc((size_t)(seq-1)*V,4);
    float *p=malloc((size_t)V*4);
    for(int t=0;t<seq-1;t++) {
        int tgt=tokens[t+1];
        const float *row=logits+t*V;
        float mx=row[0]; for(int v=1;v<V;v++) if(row[v]>mx) mx=row[v];
        float s=0; for(int v=0;v<V;v++){p[v]=expf(row[v]-mx);s+=p[v];}
        for(int v=0;v<V;v++) p[v]/=s;
        loss+=-logf(p[tgt]+1e-9f);
        for(int v=0;v<V;v++) d_logits[t*V+v]=p[v];
        d_logits[t*V+tgt]-=1.0f;
    }
    free(p); loss/=(float)(seq-1);

    float *d_lmh    =calloc((size_t)V*TF_EMBED,4);  /* heap — evita stack overflow */
    float *d_emb_arr=calloc((size_t)V*TF_EMBED,4);
    float d_final_h[TF_CONTEXT][TF_EMBED]; memset(d_final_h,0,sizeof(d_final_h));

    for(int t=0;t<seq-1;t++)
        for(int v=0;v<V;v++) {
            float dl=d_logits[t*V+v];
            for(int d=0;d<TF_EMBED;d++){
                d_lmh[v*TF_EMBED+d]+=cache->final_h[t][d]*dl;
                d_final_h[t][d]+=m->lm_head[v*TF_EMBED+d]*dl;
            }
        }
    free(d_logits);

    static float d_block_in[TF_CONTEXT][TF_EMBED];
    const float (*dout)[TF_EMBED]=(const float(*)[TF_EMBED])d_final_h;
    for(int l=TF_LAYERS-1;l>=0;l--) {
        block_backward(&m->blocks[l],seq,&cache->bc[l],
                       dout,d_block_in,lr,m->adam_t);
        dout=(const float(*)[TF_EMBED])d_block_in;
    }

    float d_pos[TF_CONTEXT*TF_EMBED]; memset(d_pos,0,sizeof(d_pos));
    for(int t=0;t<seq;t++) {
        int tok=tokens[t];
        for(int d=0;d<TF_EMBED;d++){
            d_emb_arr[tok*TF_EMBED+d]+=d_block_in[t][d];
            d_pos[t*TF_EMBED+d]+=d_block_in[t][d];
        }
    }
    adam_step(&m->a_lmh,m->lm_head,d_lmh,    V*TF_EMBED,         lr,m->adam_t);
    adam_step(&m->a_emb, m->emb,    d_emb_arr, V*TF_EMBED,         lr,m->adam_t);
    adam_step(&m->a_pos, m->pos_emb,d_pos,     TF_CONTEXT*TF_EMBED,lr,m->adam_t);
    free(d_lmh); free(d_emb_arr);
    return loss;
}

/* ── Treino ──────────────────────────────────────────────────────────────── */
void tf_train(TransformerLM *m, const int *tokens, int n,
              int epochs, float lr, int verbose) {
    int seq=TF_CONTEXT;
    if(n<seq+1){fprintf(stderr,"corpus muito pequeno\n");return;}
    TFCache *cache=calloc(1,sizeof(TFCache));
    for(int ep=0;ep<epochs;ep++) {
        float total=0; int count=0;
        for(int i=0;i+seq<n;i+=seq) {
            float *logits=tf_forward(m,tokens+i,seq,cache);
            float loss=tf_backward(m,cache,logits,tokens+i,seq,lr);
            free(logits); total+=loss; count++;
        }
        if(verbose){
            float avg=total/(float)count;
            printf("  epoch %3d/%d  loss=%.4f  ppl=%.1f\n",
                   ep+1,epochs,avg,expf(avg));
            fflush(stdout);
        }
    }
    free(cache);
}

/* ── Geração com top-p e repetition penalty ─────────────────────────────── */
static int cmp_float_desc(const void *a, const void *b) {
    float fa=*(const float*)a, fb=*(const float*)b;
    return (fb>fa)-(fb<fa);
}

void tf_generate(const TransformerLM *m, const TFVocab *v,
                 const int *seed, int seed_len, int steps,
                 const TFGenConfig *cfg) {
    int ctx[TF_CONTEXT];
    int fill=seed_len<TF_CONTEXT?seed_len:TF_CONTEXT;
    for(int i=0;i<TF_CONTEXT-fill;i++) ctx[i]=0;
    for(int i=0;i<fill;i++) ctx[TF_CONTEXT-fill+i]=seed[i];

    /* imprimir seed */
    for(int i=TF_CONTEXT-fill;i<TF_CONTEXT;i++)
        if(ctx[i]>0&&ctx[i]<v->size) printf("%s ",v->by_id[ctx[i]]->word);

    TFCache *cache=calloc(1,sizeof(TFCache));
    int V=m->vocab_size;
    float *sorted=malloc((size_t)V*sizeof(float));

    for(int step=0;step<steps;step++) {
        float *logits=tf_forward(m,ctx,TF_CONTEXT,cache);
        float *row=logits+(TF_CONTEXT-1)*V;

        /* 1. Repetition penalty nos logits (antes do softmax) */
        if(cfg->rep_penalty>1.0f)
            for(int i=0;i<TF_CONTEXT;i++) {
                int tok=ctx[i];
                if(tok<0||tok>=V) continue;
                row[tok]=(row[tok]>0)?row[tok]/cfg->rep_penalty
                                     :row[tok]*cfg->rep_penalty;
            }

        /* 2. Temperature + softmax */
        float mx=row[0]; for(int i=1;i<V;i++) if(row[i]>mx) mx=row[i];
        float s=0;
        for(int i=0;i<V;i++){row[i]=expf((row[i]-mx)/cfg->temperature);s+=row[i];}
        for(int i=0;i<V;i++) row[i]/=s;

        /* 3. Top-p (nucleus) sampling */
        if(cfg->top_p<0.9999f) {
            memcpy(sorted,row,(size_t)V*sizeof(float));
            qsort(sorted,V,sizeof(float),cmp_float_desc);
            float cum=0, thr=0;
            for(int i=0;i<V;i++){cum+=sorted[i];if(cum>=cfg->top_p){thr=sorted[i];break;}}
            float ns=0;
            for(int i=0;i<V;i++){if(row[i]<thr)row[i]=0;else ns+=row[i];}
            if(ns>0) for(int i=0;i<V;i++) row[i]/=ns;
        }

        /* 4. Amostragem */
        float r=(float)rand()/RAND_MAX, cum=0;
        int next=V-1;
        for(int i=0;i<V;i++){cum+=row[i];if(r<=cum){next=i;break;}}
        free(logits);

        /* 5. Stop token */
        if(cfg->stop_id>=0&&next==cfg->stop_id) break;

        if(next<v->size) printf("%s ",v->by_id[next]->word);
        memmove(ctx,ctx+1,(TF_CONTEXT-1)*sizeof(int));
        ctx[TF_CONTEXT-1]=next;
    }
    free(sorted); free(cache);
}
