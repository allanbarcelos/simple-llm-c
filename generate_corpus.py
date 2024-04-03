"""
generate_corpus.py
Generates a large Brazilian Portuguese training corpus for a Markov chain chatbot.
Output: corpus_pt_BR.txt (~TARGET_MB megabytes)
No external dependencies — only Python stdlib.
"""

import random
import sys
import os

# ── Configuration ────────────────────────────────────────────────────────────
TARGET_MB = 100
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus_pt_BR.txt")
PROGRESS_INTERVAL_MB = 10
ENCODING = "utf-8"
SEED = 42
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)

# ── Vocabulary ────────────────────────────────────────────────────────────────

PESSOAS = [
    "o homem", "a mulher", "a criança", "o jovem", "a jovem",
    "o pai", "a mãe", "o avô", "a avó", "o irmão", "a irmã",
    "o amigo", "a amiga", "o vizinho", "a vizinha", "o professor",
    "a professora", "o médico", "a médica", "o artista", "a artista",
    "o trabalhador", "a trabalhadora", "o estudante", "a estudante",
    "o filósofo", "a filósofa", "o poeta", "a poeta", "o músico",
    "a músca", "o cientista", "a cientista", "o jornalista", "a jornalista",
    "o político", "a política",
]

VERBOS_INF = [
    "amar", "viver", "sonhar", "aprender", "ensinar", "construir",
    "destruir", "criar", "correr", "caminhar", "falar", "escutar",
    "pensar", "sentir", "acreditar", "duvidar", "esperar", "temer",
    "celebrar", "lamentar", "buscar", "encontrar", "perder", "ganhar",
    "lutar", "desistir", "crescer", "mudar", "permanecer", "partir",
    "voltar", "descobrir", "compreender", "admirar", "criticar",
    "respeitar", "ajudar", "confiar", "mentir", "revelar",
]

# conjugated forms: (3rd person sing present, 3rd person sing past-perf, 3rd person sing future)
VERBOS_CONJ = [
    ("ama", "amou", "vai amar"),
    ("vive", "viveu", "vai viver"),
    ("sonha", "sonhou", "vai sonhar"),
    ("aprende", "aprendeu", "vai aprender"),
    ("ensina", "ensinou", "vai ensinar"),
    ("constrói", "construiu", "vai construir"),
    ("cria", "criou", "vai criar"),
    ("corre", "correu", "vai correr"),
    ("caminha", "caminhou", "vai caminhar"),
    ("fala", "falou", "vai falar"),
    ("escuta", "escutou", "vai escutar"),
    ("pensa", "pensou", "vai pensar"),
    ("sente", "sentiu", "vai sentir"),
    ("acredita", "acreditou", "vai acreditar"),
    ("duvida", "duvidou", "vai duvidar"),
    ("espera", "esperou", "vai esperar"),
    ("teme", "temeu", "vai temer"),
    ("celebra", "celebrou", "vai celebrar"),
    ("lamenta", "lamentou", "vai lamentar"),
    ("busca", "buscou", "vai buscar"),
    ("encontra", "encontrou", "vai encontrar"),
    ("perde", "perdeu", "vai perder"),
    ("ganha", "ganhou", "vai ganhar"),
    ("luta", "lutou", "vai lutar"),
    ("desiste", "desistiu", "vai desistir"),
    ("cresce", "cresceu", "vai crescer"),
    ("muda", "mudou", "vai mudar"),
    ("permanece", "permaneceu", "vai permanecer"),
    ("parte", "partiu", "vai partir"),
    ("volta", "voltou", "vai voltar"),
    ("descobre", "descobriu", "vai descobrir"),
    ("compreende", "compreendeu", "vai compreender"),
    ("admira", "admirou", "vai admirar"),
    ("critica", "criticou", "vai criticar"),
    ("respeita", "respeitou", "vai respeitar"),
    ("ajuda", "ajudou", "vai ajudar"),
    ("confia", "confiou", "vai confiar"),
    ("mente", "mentiu", "vai mentir"),
    ("revela", "revelou", "vai revelar"),
]

OBJETOS = [
    "o livro", "a carta", "a palavra", "o sonho", "a esperança",
    "o caminho", "a verdade", "a mentira", "o segredo", "o destino",
    "a história", "o futuro", "o passado", "a memória", "o silêncio",
    "a voz", "a luz", "a sombra", "o tempo", "o espaço",
    "a ideia", "o plano", "a promessa", "o erro", "a solução",
    "o problema", "a resposta", "a pergunta", "o desejo", "a realidade",
    "a ilusão", "o poder", "a fraqueza", "a coragem", "o medo",
]

LUGARES = [
    "na cidade", "no campo", "na floresta", "na praia", "nas montanhas",
    "no deserto", "na escola", "no trabalho", "em casa", "no hospital",
    "no mercado", "na praça", "no rio", "no oceano", "no céu",
    "na terra", "no jardim", "na rua", "no bairro", "no país",
    "no mundo", "no universo", "na aldeia", "no palácio", "na prisão",
    "no museu", "na biblioteca", "no teatro", "no estádio", "na fazenda",
    "no barco", "na ilha", "no vale", "na caverna", "no castelo",
]

TEMPO = [
    "hoje", "ontem", "amanhã", "agora", "logo",
    "sempre", "nunca", "às vezes", "frequentemente", "raramente",
    "de manhã", "à tarde", "à noite", "ao amanhecer", "ao entardecer",
    "neste momento", "no passado", "no futuro", "durante anos", "por um instante",
    "com o tempo", "desde criança", "até o fim", "no início", "no final",
    "de repente", "aos poucos", "cada vez mais", "de vez em quando", "por muito tempo",
]

EMOCOES = [
    "feliz", "triste", "ansioso", "calmo", "apaixonado",
    "frustrado", "esperançoso", "desesperado", "curioso", "entediado",
    "orgulhoso", "envergonhado", "grato", "ressentido", "encantado",
    "assustado", "corajoso", "inseguro", "confiante", "surpreso",
    "aliviado", "nostálgico", "eufórico", "melancólico", "sereno",
    "irritado", "compassivo", "indiferente", "apático", "entusiasmado",
]

CONCEITOS = [
    "a liberdade", "a justiça", "a paz", "a guerra", "a beleza",
    "a sabedoria", "a ignorância", "o amor", "o ódio", "a bondade",
    "a maldade", "a verdade", "a mentira", "a fé", "a dúvida",
    "o conhecimento", "a arte", "a ciência", "a religião", "a filosofia",
    "a cultura", "a tradição", "a inovação", "a democracia", "a tirania",
    "a igualdade", "a desigualdade", "o progresso", "a decadência", "a eternidade",
    "o efêmero", "o absoluto", "o relativo", "o caos", "a ordem",
]

NATUREZA = [
    "o sol", "a lua", "as estrelas", "o vento", "a chuva",
    "a neve", "o fogo", "a água", "a terra", "o ar",
    "a árvore", "a flor", "o rio", "o mar", "a montanha",
    "o pássaro", "o lobo", "o leão", "a borboleta", "a abelha",
    "o carvalho", "o bambu", "a rosa", "o espinho", "a raiz",
    "o fruto", "a semente", "a tempestade", "o arco-íris", "a neblina",
    "o vulcão", "o iceberg", "o deserto", "a savana", "a tundra",
]

COMIDA = [
    "o pão", "o arroz", "o feijão", "a carne", "o peixe",
    "a fruta", "o legume", "o queijo", "o mel", "o café",
    "o vinho", "a água", "o suco", "a sopa", "o bolo",
    "a farofa", "o churrasco", "a feijoada", "o acarajé", "o tapioca",
    "a manga", "o caju", "a goiaba", "o açaí", "o milho",
    "a mandioca", "o inhame", "o coco", "o dendê", "o quiabo",
    "a pimenta", "o alho", "a cebola", "o tomate", "o limão",
]

TRABALHO = [
    "o emprego", "o salário", "a carreira", "o projeto", "a reunião",
    "o prazo", "a tarefa", "o colega", "o chefe", "a empresa",
    "o negócio", "o lucro", "o prejuízo", "a meta", "o resultado",
    "a profissão", "o ofício", "a habilidade", "o talento", "o esforço",
    "a dedicação", "a competência", "o fracasso", "o sucesso", "a oportunidade",
    "o desafio", "a inovação", "a tecnologia", "a ferramenta", "o contrato",
    "a parceria", "o mercado", "o cliente", "o produto", "o serviço",
]

RELACIONAMENTOS = [
    "a amizade", "o amor", "o casamento", "a família", "a rivalidade",
    "a parceria", "a cumplicidade", "a traição", "o perdão", "a confiança",
    "o respeito", "a admiração", "o ciúme", "a lealdade", "a distância",
    "o encontro", "a despedida", "o reencontro", "o conflito", "a reconciliação",
    "a solidariedade", "a compaixão", "a generosidade", "o egoísmo", "o sacrifício",
    "a dedicação", "o cuidado", "o abandono", "a saudade", "o afeto",
]

ADVERBIOS = [
    "muito", "pouco", "bastante", "demais", "profundamente",
    "claramente", "facilmente", "dificilmente", "sinceramente", "certamente",
    "provavelmente", "felizmente", "infelizmente", "surpreendentemente", "gradualmente",
    "repentinamente", "eternamente", "brevemente", "silenciosamente", "corajosamente",
]

ADJETIVOS_GERAIS = [
    "grande", "pequeno", "forte", "fraco", "belo",
    "feio", "rico", "pobre", "sábio", "tolo",
    "rápido", "lento", "antigo", "moderno", "simples",
    "complexo", "claro", "obscuro", "verdadeiro", "falso",
    "importante", "irrelevante", "necessário", "desnecessário", "possível",
    "impossível", "visível", "invisível", "eterno", "temporário",
]

CONJUNCOES = [
    "porque", "portanto", "assim", "então", "contudo",
    "porém", "mas", "e", "ou", "enquanto",
    "embora", "apesar de", "desde que", "para que", "antes que",
]

PRONOMES = [
    "ele", "ela", "nós", "eles", "elas",
    "eu", "você", "alguém", "ninguém", "todos",
    "cada um", "muitos", "poucos", "alguns", "outros",
]

# ── Helper functions ──────────────────────────────────────────────────────────

def rp(lst):
    """Random pick from list."""
    return random.choice(lst)

def rvc():
    """Return a random (present, past, future) verb tuple."""
    return random.choice(VERBOS_CONJ)

def cap(s):
    """Capitalize first character."""
    return s[0].upper() + s[1:] if s else s

def sent(text):
    """End with period if no final punctuation."""
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    return cap(text)

def question(text):
    text = text.strip()
    if text and text[-1] not in "!?.":
        text += "?"
    return cap(text)

# ── Sentence templates ────────────────────────────────────────────────────────
# Each function returns one sentence (str, no trailing newline).

def tmpl_statement_simple():
    pessoa = rp(PESSOAS)
    v_pres, _, _ = rvc()
    obj = rp(OBJETOS)
    lugar = rp(LUGARES)
    return sent(f"{pessoa} {v_pres} {obj} {lugar}")

def tmpl_statement_past():
    pessoa = rp(PESSOAS)
    _, v_past, _ = rvc()
    obj = rp(OBJETOS)
    tempo = rp(TEMPO)
    return sent(f"{pessoa} {v_past} {obj} {tempo}")

def tmpl_statement_future():
    pessoa = rp(PESSOAS)
    _, _, v_fut = rvc()
    obj = rp(OBJETOS)
    lugar = rp(LUGARES)
    return sent(f"{pessoa} {v_fut} {obj} {lugar}")

def tmpl_emotion_statement():
    pessoa = rp(PESSOAS)
    emocao = rp(EMOCOES)
    v_pres, _, _ = rvc()
    obj = rp(OBJETOS)
    return sent(f"{pessoa} está {emocao} e {v_pres} {obj}")

def tmpl_nature_metaphor():
    nat = rp(NATUREZA)
    adv = rp(ADVERBIOS)
    adj = rp(ADJETIVOS_GERAIS)
    conceito = rp(CONCEITOS)
    return sent(f"{nat} é {adv} {adj}, assim como {conceito}")

def tmpl_question_simple():
    pessoa = rp(PESSOAS)
    v_pres, _, _ = rvc()
    obj = rp(OBJETOS)
    return question(f"Por que {pessoa} {v_pres} {obj}")

def tmpl_question_where():
    pessoa = rp(PESSOAS)
    obj = rp(OBJETOS)
    return question(f"Onde {pessoa} encontrou {obj}")

def tmpl_question_when():
    pessoa = rp(PESSOAS)
    v_inf = rp(VERBOS_INF)
    obj = rp(OBJETOS)
    return question(f"Quando {pessoa} vai {v_inf} {obj}")

def tmpl_question_how():
    pessoa = rp(PESSOAS)
    v_inf = rp(VERBOS_INF)
    conceito = rp(CONCEITOS)
    return question(f"Como {pessoa} pode {v_inf} {conceito}")

def tmpl_question_what():
    pessoa = rp(PESSOAS)
    emocao = rp(EMOCOES)
    return question(f"O que faz {pessoa} se sentir {emocao}")

def tmpl_conditional_if():
    pessoa = rp(PESSOAS)
    v_inf = rp(VERBOS_INF)
    obj = rp(OBJETOS)
    pessoa2 = rp(PESSOAS)
    v_inf2 = rp(VERBOS_INF)
    conceito = rp(CONCEITOS)
    return sent(f"Se {pessoa} puder {v_inf} {obj}, então {pessoa2} vai {v_inf2} {conceito}")

def tmpl_conditional_when():
    pessoa = rp(PESSOAS)
    v_inf = rp(VERBOS_INF)
    conceito = rp(CONCEITOS)
    adv = rp(ADVERBIOS)
    v_inf2 = rp(VERBOS_INF)
    obj = rp(OBJETOS)
    return sent(f"Quando {pessoa} decidir {v_inf} {conceito}, {adv} vai {v_inf2} {obj}")

def tmpl_comparison():
    pessoa = rp(PESSOAS)
    adj = rp(ADJETIVOS_GERAIS)
    pessoa2 = rp(PESSOAS)
    adv = rp(ADVERBIOS)
    return sent(f"{pessoa} é tão {adj} quanto {pessoa2}, mas {adv} mais corajoso")

def tmpl_comparison_conceito():
    conceito1 = rp(CONCEITOS)
    adv = rp(ADVERBIOS)
    adj = rp(ADJETIVOS_GERAIS)
    conceito2 = rp(CONCEITOS)
    return sent(f"{conceito1} é {adv} mais {adj} do que {conceito2}")

def tmpl_proverb_style():
    nat = rp(NATUREZA)
    v_pres, _, _ = rvc()
    obj = rp(OBJETOS)
    v_pres2, _, _ = rvc()
    conceito = rp(CONCEITOS)
    return sent(f"Quem {v_pres} {obj} com sabedoria, {v_pres2} {conceito} com o coração")

def tmpl_proverb_style2():
    nat = rp(NATUREZA)
    adj = rp(ADJETIVOS_GERAIS)
    nat2 = rp(NATUREZA)
    adv = rp(ADVERBIOS)
    return sent(f"Assim como {nat} é {adj}, {nat2} também é {adv} necessário")

def tmpl_opinion():
    pronome = rp(PRONOMES)
    v_inf = rp(VERBOS_INF)
    conceito = rp(CONCEITOS)
    adv = rp(ADVERBIOS)
    adj = rp(ADJETIVOS_GERAIS)
    return sent(f"Para mim, {v_inf} {conceito} é {adv} {adj}")

def tmpl_opinion2():
    pronome = rp(PRONOMES)
    adj = rp(ADJETIVOS_GERAIS)
    v_inf = rp(VERBOS_INF)
    obj = rp(OBJETOS)
    lugar = rp(LUGARES)
    return sent(f"Acredito que é {adj} {v_inf} {obj} {lugar}")

def tmpl_description_pessoa():
    pessoa = rp(PESSOAS)
    adj1 = rp(ADJETIVOS_GERAIS)
    adj2 = rp(EMOCOES)
    v_pres, _, _ = rvc()
    conceito = rp(CONCEITOS)
    return sent(f"{pessoa} é {adj1} e {adj2}, e sempre {v_pres} {conceito}")

def tmpl_description_lugar():
    lugar = rp(LUGARES)
    adj = rp(ADJETIVOS_GERAIS)
    nat = rp(NATUREZA)
    adv = rp(ADVERBIOS)
    return sent(f"{lugar}, onde {nat} é {adv} {adj} e o tempo passa devagar")

def tmpl_cause_effect():
    pessoa = rp(PESSOAS)
    v_past, v_pres_aux = rvc()[1], rvc()[0]
    obj = rp(OBJETOS)
    conj = rp(["por isso", "portanto", "assim", "então"])
    pessoa2 = rp(PESSOAS)
    v_inf = rp(VERBOS_INF)
    conceito = rp(CONCEITOS)
    return sent(f"{pessoa} {v_past} {obj}, {conj} {pessoa2} precisou {v_inf} {conceito}")

def tmpl_cause_effect2():
    conceito = rp(CONCEITOS)
    v_pres, _, _ = rvc()
    obj = rp(OBJETOS)
    conj = rp(["porque", "pois", "visto que", "já que"])
    pessoa = rp(PESSOAS)
    adv = rp(ADVERBIOS)
    v_inf = rp(VERBOS_INF)
    return sent(f"{conceito} {v_pres} {obj} {conj} {pessoa} {adv} decidiu {v_inf}")

def tmpl_food_culture():
    comida = rp(COMIDA)
    adv = rp(ADVERBIOS)
    adj = rp(ADJETIVOS_GERAIS)
    lugar = rp(LUGARES)
    return sent(f"{comida} é {adv} {adj} e faz parte da vida {lugar}")

def tmpl_work_life():
    trabalho_item = rp(TRABALHO)
    v_pres, _, _ = rvc()
    pessoa = rp(PESSOAS)
    emocao = rp(EMOCOES)
    adv = rp(ADVERBIOS)
    return sent(f"{trabalho_item} {v_pres} {pessoa}, que se sente {emocao} {adv}")

def tmpl_relationship():
    rel = rp(RELACIONAMENTOS)
    v_pres, _, _ = rvc()
    conceito = rp(CONCEITOS)
    adv = rp(ADVERBIOS)
    pessoa = rp(PESSOAS)
    return sent(f"{rel} {v_pres} {conceito} de forma {adv} quando {pessoa} está presente")

def tmpl_time_reflection():
    tempo = rp(TEMPO)
    pessoa = rp(PESSOAS)
    v_pres, v_past, _ = rvc()
    obj = rp(OBJETOS)
    conceito = rp(CONCEITOS)
    return sent(f"{tempo}, {pessoa} {v_pres} {obj} e lembra de {conceito}")

def tmpl_negation():
    pessoa = rp(PESSOAS)
    v_pres, _, _ = rvc()
    obj = rp(OBJETOS)
    adv = rp(ADVERBIOS)
    return sent(f"{pessoa} não {v_pres} {obj}, pois nunca foi {adv} simples")

def tmpl_exclamation():
    adv = rp(ADVERBIOS)
    adj = rp(ADJETIVOS_GERAIS)
    nat = rp(NATUREZA)
    conceito = rp(CONCEITOS)
    text = f"Que {adv} {adj} é {nat} diante de {conceito}!"
    return cap(text)

def tmpl_abstract_reflection():
    conceito1 = rp(CONCEITOS)
    conj = rp(CONJUNCOES)
    conceito2 = rp(CONCEITOS)
    v_pres, _, _ = rvc()
    obj = rp(OBJETOS)
    return sent(f"{conceito1} existe {conj} {conceito2} também {v_pres} {obj}")

def tmpl_nature_observation():
    nat = rp(NATUREZA)
    v_pres, _, _ = rvc()
    lugar = rp(LUGARES)
    adv = rp(ADVERBIOS)
    adj = rp(ADJETIVOS_GERAIS)
    return sent(f"{nat} {v_pres} {lugar} de forma {adv} {adj}")

def tmpl_dialogue_style():
    pessoa = rp(PESSOAS)
    v_inf = rp(VERBOS_INF)
    obj = rp(OBJETOS)
    pessoa2 = rp(PESSOAS)
    return sent(f"{pessoa} disse que precisa {v_inf} {obj} antes que {pessoa2} chegue")

def tmpl_philosophical():
    v_inf = rp(VERBOS_INF)
    conceito = rp(CONCEITOS)
    adv = rp(ADVERBIOS)
    v_inf2 = rp(VERBOS_INF)
    obj = rp(OBJETOS)
    return sent(f"Saber {v_inf} {conceito} é {adv} mais importante do que {v_inf2} {obj}")

def tmpl_contrast():
    pessoa = rp(PESSOAS)
    emocao1 = rp(EMOCOES)
    conj = rp(["mas", "porém", "contudo", "embora"])
    emocao2 = rp(EMOCOES)
    v_pres, _, _ = rvc()
    obj = rp(OBJETOS)
    return sent(f"{pessoa} estava {emocao1}, {conj} continuou {emocao2} e {v_pres} {obj}")

def tmpl_list_style():
    pessoas = random.sample(PESSOAS, 2)
    v_inf1 = rp(VERBOS_INF)
    v_inf2 = rp(VERBOS_INF)
    obj = rp(OBJETOS)
    conceito = rp(CONCEITOS)
    return sent(f"{pessoas[0]} e {pessoas[1]} decidiram {v_inf1} {obj} e {v_inf2} {conceito}")

def tmpl_generalization():
    pronome = rp(["todo mundo", "ninguém", "a maioria das pessoas", "poucos", "muitos"])
    v_pres, _, _ = rvc()
    conceito = rp(CONCEITOS)
    adv = rp(ADVERBIOS)
    obj = rp(OBJETOS)
    return sent(f"{pronome} {v_pres} {conceito} de forma {adv}, buscando {obj}")

def tmpl_sensory():
    nat = rp(NATUREZA)
    adj = rp(ADJETIVOS_GERAIS)
    adv = rp(ADVERBIOS)
    lugar = rp(LUGARES)
    tempo = rp(TEMPO)
    return sent(f"{nat} {adj} toca {adv} a alma de quem vive {lugar} {tempo}")

def tmpl_memory():
    pessoa = rp(PESSOAS)
    tempo = rp(TEMPO)
    v_past, _, _ = (rvc()[1], None, None)
    obj = rp(OBJETOS)
    v_inf = rp(VERBOS_INF)
    conceito = rp(CONCEITOS)
    return sent(f"{pessoa} lembra que {tempo} {v_past} {obj} e aprendeu a {v_inf} {conceito}")

# ── Template registry ─────────────────────────────────────────────────────────

TEMPLATES = [
    tmpl_statement_simple,
    tmpl_statement_past,
    tmpl_statement_future,
    tmpl_emotion_statement,
    tmpl_nature_metaphor,
    tmpl_question_simple,
    tmpl_question_where,
    tmpl_question_when,
    tmpl_question_how,
    tmpl_question_what,
    tmpl_conditional_if,
    tmpl_conditional_when,
    tmpl_comparison,
    tmpl_comparison_conceito,
    tmpl_proverb_style,
    tmpl_proverb_style2,
    tmpl_opinion,
    tmpl_opinion2,
    tmpl_description_pessoa,
    tmpl_description_lugar,
    tmpl_cause_effect,
    tmpl_cause_effect2,
    tmpl_food_culture,
    tmpl_work_life,
    tmpl_relationship,
    tmpl_time_reflection,
    tmpl_negation,
    tmpl_exclamation,
    tmpl_abstract_reflection,
    tmpl_nature_observation,
    tmpl_dialogue_style,
    tmpl_philosophical,
    tmpl_contrast,
    tmpl_list_style,
    tmpl_generalization,
    tmpl_sensory,
    tmpl_memory,
]

# ── Main generation loop ──────────────────────────────────────────────────────

def generate_sentence():
    fn = random.choice(TEMPLATES)
    return fn()

def main():
    target_bytes = TARGET_MB * 1024 * 1024
    progress_threshold = PROGRESS_INTERVAL_MB * 1024 * 1024
    next_progress = progress_threshold

    print(f"Generating {TARGET_MB}MB corpus -> {OUTPUT_FILE}", file=sys.stderr)

    bytes_written = 0
    with open(OUTPUT_FILE, "w", encoding=ENCODING) as fout:
        while bytes_written < target_bytes:
            # Write a paragraph (5-12 sentences) at a time for natural flow
            para_len = random.randint(5, 12)
            lines = []
            for _ in range(para_len):
                lines.append(generate_sentence())
            paragraph = " ".join(lines) + "\n\n"
            encoded = paragraph.encode(ENCODING)
            fout.write(paragraph)
            bytes_written += len(encoded)

            if bytes_written >= next_progress:
                mb_done = bytes_written / (1024 * 1024)
                pct = bytes_written / target_bytes * 100
                print(f"  {mb_done:.1f} MB written ({pct:.1f}%)", file=sys.stderr)
                next_progress += progress_threshold

    final_mb = bytes_written / (1024 * 1024)
    print(f"Done. {final_mb:.2f} MB written to {OUTPUT_FILE}", file=sys.stderr)

if __name__ == "__main__":
    main()
