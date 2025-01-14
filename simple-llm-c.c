/*
  Description: C program for generating chat responses using Markov Chain
  Uses a hash table for O(1) transition lookup — scales to large corpora.
  Author: Allan Barcelos <allan@barcelos.dev>
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>

#define MAX_LEN       100
#define MAX_LINE_LEN  1024
#define MAX_TOKENS    500
#define ORDER         2
#define MAX_GENERATED 20
#define CONTEXT_LEN   (ORDER * (MAX_LEN + 1))
#define HASH_SIZE     (1 << 18)   /* 262,144 buckets — ~2 MB pointer array */

/* Candidate next word in a transition */
typedef struct WordNode {
    char word[MAX_LEN];
    struct WordNode *next;
} WordNode;

/* One transition: context string -> list of possible next words */
typedef struct Transition {
    char context[CONTEXT_LEN];
    WordNode *candidates;
    int count;
    struct Transition *next;    /* next in hash bucket chain */
} Transition;

/* Markov chain backed by a hash table */
typedef struct {
    Transition **buckets;       /* HASH_SIZE bucket heads                */
    int transition_count;
} MarkovChain;

static MarkovChain *markovChain;

/* --- Utilities --- */

static unsigned long hash_str(const char *s) {
    unsigned long h = 5381;
    int c;
    while ((c = (unsigned char)*s++))
        h = ((h << 5) + h) + c;    /* djb2 */
    return h;
}

static void normalize_word(char *word) {
    size_t start = 0;
    while (word[start] && ispunct((unsigned char)word[start]))
        start++;
    if (start > 0)
        memmove(word, word + start, strlen(word + start) + 1);

    size_t len = strlen(word);
    while (len > 0 && ispunct((unsigned char)word[len - 1]))
        word[--len] = '\0';

    for (size_t i = 0; word[i]; i++) {
        if ((unsigned char)word[i] >= 'A' && (unsigned char)word[i] <= 'Z')
            word[i] += 'a' - 'A';
    }
}

static void strip_newline(char *s) {
    size_t len = strlen(s);
    if (len > 0 && (s[len - 1] == '\n' || s[len - 1] == '\r'))
        s[len - 1] = '\0';
    len = strlen(s);
    if (len > 0 && s[len - 1] == '\r')
        s[len - 1] = '\0';
}

static void flush_stdin(void) {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}

static void build_context(char *out, char **words, int n) {
    out[0] = '\0';
    for (int i = 0; i < n; i++) {
        if (i > 0)
            strncat(out, " ", CONTEXT_LEN - strlen(out) - 1);
        strncat(out, words[i], CONTEXT_LEN - strlen(out) - 1);
    }
}

/* --- Chain management (hash table) --- */

static Transition *get_or_create_transition(const char *context) {
    unsigned long h = hash_str(context) & (HASH_SIZE - 1);

    for (Transition *t = markovChain->buckets[h]; t != NULL; t = t->next) {
        if (strcmp(t->context, context) == 0)
            return t;
    }

    Transition *t = (Transition *)malloc(sizeof(Transition));
    if (t == NULL) return NULL;

    strncpy(t->context, context, CONTEXT_LEN - 1);
    t->context[CONTEXT_LEN - 1] = '\0';
    t->candidates = NULL;
    t->count = 0;
    t->next = markovChain->buckets[h];
    markovChain->buckets[h] = t;
    markovChain->transition_count++;
    return t;
}

/* O(1) exact-key lookup used during generation */
static Transition *find_transition(const char *context) {
    unsigned long h = hash_str(context) & (HASH_SIZE - 1);
    for (Transition *t = markovChain->buckets[h]; t != NULL; t = t->next) {
        if (strcmp(t->context, context) == 0)
            return t;
    }
    return NULL;
}

static void add_candidate(Transition *t, const char *word) {
    WordNode *node = (WordNode *)malloc(sizeof(WordNode));
    if (node == NULL) return;

    strncpy(node->word, word, MAX_LEN - 1);
    node->word[MAX_LEN - 1] = '\0';
    node->next = t->candidates;
    t->candidates = node;
    t->count++;
}

/* --- Training --- */

static void train_tokens(char **tokens, int n) {
    for (int i = 0; i < n; i++) {
        int ctx_size = i < ORDER ? i : ORDER;
        char context[CONTEXT_LEN];
        build_context(context, tokens + (i - ctx_size), ctx_size);

        Transition *t = get_or_create_transition(context);
        if (t != NULL)
            add_candidate(t, tokens[i]);
    }
}

static void train_line(char *line) {
    char *tokens[MAX_TOKENS];
    int n = 0;

    char *tok = strtok(line, " \t\r\n");
    while (tok != NULL && n < MAX_TOKENS) {
        normalize_word(tok);
        if (strlen(tok) > 0)
            tokens[n++] = tok;
        tok = strtok(NULL, " \t\r\n");
    }

    if (n > 0)
        train_tokens(tokens, n);
}

void trainingFromInput(void) {
    char input[MAX_LINE_LEN];
    printf("Enter training phrases (type 'exit' to finish):\n");

    while (1) {
        printf("> ");
        if (fgets(input, sizeof(input), stdin) == NULL) break;
        strip_newline(input);

        if (strcmp(input, "exit") == 0) break;
        if (strlen(input) == 0) continue;

        train_line(input);
    }

    printf("Training completed. %d transitions.\n", markovChain->transition_count);
}

void trainingFromFile(void) {
    char filename[MAX_LEN];
    printf("Enter filename with training phrases: ");

    if (fgets(filename, sizeof(filename), stdin) == NULL) return;
    strip_newline(filename);

    if (strlen(filename) == 0) {
        printf("No filename provided.\n");
        return;
    }

    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Failed to open '%s'.\n", filename);
        return;
    }

    char line[MAX_LINE_LEN];
    int trained = 0;
    while (fgets(line, sizeof(line), file)) {
        strip_newline(line);
        if (strlen(line) == 0) continue;
        train_line(line);
        trained++;
        if (trained % 100000 == 0)
            printf("  %d lines processed...\n", trained);
    }

    fclose(file);
    printf("Trained on %d lines. %d unique transitions.\n",
           trained, markovChain->transition_count);
}

/* --- Chat / Response generation --- */

#define MAX_SEED_CANDIDATES 100

static const char *STOP_WORDS[] = {
    "o", "a", "os", "as", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "por", "para", "com", "sem", "sob", "sobre", "entre",
    "que", "se", "não", "nem", "e", "ou", "mas", "pois",
    "ao", "à", "pelo", "pela", "pelos", "pelas",
    "é", "são", "foi", "ser", "ter", "há", "seria",
    NULL
};

static int is_stop_word(const char *word) {
    for (int i = 0; STOP_WORDS[i] != NULL; i++) {
        if (strcmp(word, STOP_WORDS[i]) == 0)
            return 1;
    }
    return 0;
}

/*
 * Collect up to MAX_SEED_CANDIDATES transitions whose context contains
 * `word` as a whole word, then pick one at random.
 * Randomising avoids always returning the same (bucket-order) match.
 */
static Transition *find_transition_with_word_random(const char *word) {
    Transition *matches[MAX_SEED_CANDIDATES];
    int count = 0;
    size_t wlen = strlen(word);

    for (int b = 0; b < HASH_SIZE && count < MAX_SEED_CANDIDATES; b++) {
        for (Transition *t = markovChain->buckets[b];
             t != NULL && count < MAX_SEED_CANDIDATES; t = t->next) {
            if (t->count == 0) continue;
            const char *p = t->context;
            while ((p = strstr(p, word)) != NULL) {
                int before_ok = (p == t->context) || (*(p - 1) == ' ');
                int after_ok  = (p[wlen] == '\0')  || (p[wlen] == ' ');
                if (before_ok && after_ok) { matches[count++] = t; break; }
                p += wlen;
            }
        }
    }

    if (count == 0) return NULL;
    return matches[rand() % count];
}

static void generate_response(char **input_tokens, int n) {
    Transition *seed = NULL;

    /*
     * Strategy 0: exact context match on the user's last ORDER (or fewer)
     * words.  This is O(1) via hash lookup and gives the most on-topic seed
     * because the chain continues directly from what the user just said.
     */
    for (int ctx = (n < ORDER ? n : ORDER); ctx >= 1 && seed == NULL; ctx--) {
        char context[CONTEXT_LEN];
        build_context(context, input_tokens + (n - ctx), ctx);
        seed = find_transition(context);
    }

    /* Strategy 1: random match on content words (non-stop-words) */
    for (int i = n - 1; i >= 0 && seed == NULL; i--) {
        if (!is_stop_word(input_tokens[i]))
            seed = find_transition_with_word_random(input_tokens[i]);
    }

    /* Strategy 2: random match on any word */
    for (int i = n - 1; i >= 0 && seed == NULL; i--)
        seed = find_transition_with_word_random(input_tokens[i]);

    /* Fallback: any sentence start */
    if (seed == NULL)
        seed = find_transition("");

    if (seed == NULL || seed->count == 0) {
        printf("Bot: ...\n");
        return;
    }

    /* Pick first response word from the seed's candidates */
    int pick = rand() % seed->count;
    WordNode *w = seed->candidates;
    for (int i = 0; i < pick; i++)
        w = w->next;

    printf("Bot: %s", w->word);

    /* Build sliding window from seed context + first generated word */
    char ctx_copy[CONTEXT_LEN];
    strncpy(ctx_copy, seed->context, CONTEXT_LEN - 1);
    ctx_copy[CONTEXT_LEN - 1] = '\0';

    char *window[ORDER];
    int wsize = 0;

    char *ctx_tok = strtok(ctx_copy, " ");
    while (ctx_tok != NULL && wsize < ORDER) {
        window[wsize++] = ctx_tok;
        ctx_tok = strtok(NULL, " ");
    }

    if (wsize < ORDER) {
        window[wsize++] = w->word;
    } else {
        for (int i = 0; i < ORDER - 1; i++)
            window[i] = window[i + 1];
        window[ORDER - 1] = w->word;
    }

    /* Continue generating via hash lookups */
    for (int step = 1; step < MAX_GENERATED; step++) {
        char context[CONTEXT_LEN];
        build_context(context, window, wsize);

        Transition *t = find_transition(context);
        if (t == NULL || t->count == 0) break;

        pick = rand() % t->count;
        w = t->candidates;
        for (int i = 0; i < pick; i++)
            w = w->next;

        printf(" %s", w->word);

        if (wsize < ORDER) {
            window[wsize++] = w->word;
        } else {
            for (int i = 0; i < ORDER - 1; i++)
                window[i] = window[i + 1];
            window[ORDER - 1] = w->word;
        }
    }

    printf("\n");
}

void chat(void) {
    if (markovChain == NULL || markovChain->transition_count == 0) {
        printf("No training data. Please train first (options 1 or 2).\n");
        return;
    }

    printf("Chat started. Type 'exit' to return to menu.\n\n");

    char input[MAX_LINE_LEN];
    while (1) {
        printf("You: ");
        if (fgets(input, sizeof(input), stdin) == NULL) break;
        strip_newline(input);

        if (strlen(input) == 0) continue;
        if (strcmp(input, "exit") == 0) break;

        char buf[MAX_LINE_LEN];
        strncpy(buf, input, sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';

        char *tokens[MAX_TOKENS];
        int n = 0;
        char *tok = strtok(buf, " \t");
        while (tok != NULL && n < MAX_TOKENS) {
            normalize_word(tok);
            if (strlen(tok) > 0)
                tokens[n++] = tok;
            tok = strtok(NULL, " \t");
        }

        if (n > 0)
            generate_response(tokens, n);

        printf("\n");
    }
}

/* --- Cleanup --- */

static void free_chain(void) {
    if (markovChain == NULL) return;
    for (int b = 0; b < HASH_SIZE; b++) {
        Transition *t = markovChain->buckets[b];
        while (t != NULL) {
            WordNode *w = t->candidates;
            while (w != NULL) {
                WordNode *tmp = w;
                w = w->next;
                free(tmp);
            }
            Transition *tmp = t;
            t = t->next;
            free(tmp);
        }
    }
    free(markovChain->buckets);
    free(markovChain);
    markovChain = NULL;
}

/* --- Main --- */

int main(void) {
    srand((unsigned)time(NULL));

    markovChain = (MarkovChain *)malloc(sizeof(MarkovChain));
    if (markovChain == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }
    markovChain->buckets = (Transition **)calloc(HASH_SIZE, sizeof(Transition *));
    if (markovChain->buckets == NULL) {
        fprintf(stderr, "Hash table allocation failed.\n");
        free(markovChain);
        return 1;
    }
    markovChain->transition_count = 0;

    int choice;
    do {
        printf("Menu:\n");
        printf("1. Train from input\n");
        printf("2. Train from file\n");
        printf("3. Start chat\n");
        printf("4. Exit\n");
        printf("Enter your choice: ");

        if (scanf("%d", &choice) != 1) {
            flush_stdin();
            printf("Invalid input.\n\n");
            choice = 0;
            continue;
        }
        flush_stdin();
        printf("\n");

        switch (choice) {
            case 1: trainingFromInput(); break;
            case 2: trainingFromFile();  break;
            case 3: chat();              break;
            case 4: printf("Exiting.\n"); break;
            default: printf("Invalid choice.\n"); break;
        }

        printf("\n");
    } while (choice != 4);

    free_chain();
    return 0;
}
