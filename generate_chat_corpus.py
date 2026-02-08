"""
Gera corpus conversacional em formato "user <pergunta> assistant <resposta>"
Saída: chat_corpus.txt

Uso: python generate_chat_corpus.py
"""
import random

random.seed(42)

# ── Pares fixos — provérbios e significados ───────────────────────────────
proverbios = [
    ("quem ferro fere com ferro sera ferido", "quem faz o mal ao outro recebe o mesmo de volta mais cedo ou mais tarde"),
    ("mais vale tarde do que nunca", "e melhor agir com atraso do que nunca agir pois toda acao tem valor"),
    ("quem nao arrisca nao petisca", "sem coragem e disposicao para tentar nao se alcanca nada de valor"),
    ("agua mole em pedra dura tanto bate ate que fura", "a persistencia e a constancia vencem qualquer obstáculo por maior que seja"),
    ("em terra de cego quem tem um olho e rei", "numa situacao de pouca competencia quem tem alguma habilidade se destaca"),
    ("quem com ferro fere com ferro sera ferido", "as consequencias de nossas acoes retornam para nos mesmos"),
    ("o habito nao faz o monge", "a aparencia nao revela o carater de uma pessoa"),
    ("nao deixe para amanha o que pode fazer hoje", "agir no presente e melhor que adiar pois o tempo e precioso"),
    ("devagar se vai ao longe", "a paciencia e a calma levam ao sucesso melhor que a pressa"),
    ("quem semeia vento colhe tempestade", "quem pratica o mal cria condicoes para sofrer consequencias graves"),
    ("a ocasiao faz o ladrao", "as circunstancias e oportunidades influenciam as escolhas das pessoas"),
    ("nao ha mal que sempre dure nem bem que nunca se acabe", "tudo na vida e passageiro tanto o sofrimento quanto a alegria"),
    ("quem espera sempre alcanca", "a paciencia e a perseveranca sao recompensadas com o tempo"),
    ("de grao em grao a galinha enche o papo", "pequenas acoes somadas produzem grandes resultados"),
    ("quem tem boca vai a roma", "quem pergunta e busca informacao alcanca seu destino"),
    ("a mentira tem pernas curtas", "a mentira nao se sustenta por muito tempo e a verdade sempre aparece"),
    ("olho por olho dente por dente", "cada acao gera uma reacao equivalente no mundo"),
    ("quem procura acha", "o esforco e a busca sao recompensados com descobertas e resultados"),
    ("o segredo e a alma do negocio", "manter informacoes importantes em sigilo e fundamental para o sucesso"),
    ("nao adianta chorar sobre o leite derramado", "nao e util se arrepender do que ja passou o importante e agir agora"),
    ("cada macaco no seu galho", "cada pessoa deve cuidar do seu proprio espaco e responsabilidades"),
    ("quem ri por ultimo ri melhor", "a vitoria final e mais importante do que vantagens temporarias"),
    ("casa de ferreiro espeto de pau", "e comum negligenciar o proprio ambiente enquanto cuida do dos outros"),
    ("o pior cego e o que nao quer ver", "a ignorancia escolhida e pior que a ignorancia involuntaria"),
    ("uma andorinha so nao faz verao", "um esforco isolado nao e suficiente para grandes realizacoes"),
]

# ── Perguntas gerais sobre a vida ─────────────────────────────────────────
vida = [
    ("o que e amor", "amor e um sentimento profundo de afeto e cuidado que une as pessoas e da sentido a vida"),
    ("o que e felicidade", "felicidade e um estado de bem estar e satisfacao que vem de dentro e das relacoes que cultivamos"),
    ("como ser feliz", "cultivar relacoes positivas ter proposito e agradecer pelo que se tem sao caminhos para a felicidade"),
    ("o que e amizade", "amizade e um laco de confianca respeito e afeto que se constroi com o tempo e dedicacao"),
    ("como ter uma vida boa", "uma vida boa vem do equilibrio entre trabalho descanso relacionamentos e proposito"),
    ("o que e coragem", "coragem e a capacidade de agir apesar do medo e das dificuldades que a vida apresenta"),
    ("como superar dificuldades", "enfrentar os problemas com calma buscar apoio e aprender com cada experiencia ajuda a superar as dificuldades"),
    ("o que e sucesso", "sucesso e alcancar os proprios objetivos e viver de acordo com os proprios valores"),
    ("como ter disciplina", "estabelecer rotinas criar habitos saudaveis e manter o foco nos objetivos sao formas de desenvolver disciplina"),
    ("o que faz uma pessoa ser boa", "empatia honestidade generosidade e respeito ao proximo sao qualidades de uma pessoa boa"),
    ("qual o sentido da vida", "o sentido da vida e construido por cada pessoa atraves de suas escolhas relacoes e propositos"),
    ("como lidar com o fracasso", "ver o fracasso como aprendizado e nao como derrota definitiva e o caminho para crescer"),
    ("o que e sabedoria", "sabedoria e a capacidade de usar o conhecimento e a experiencia para tomar boas decisoes"),
    ("como ser mais produtivo", "organizar as tarefas eliminar distraccoes e descansar adequadamente aumentam a produtividade"),
    ("o que e persistencia", "persistencia e continuar tentando mesmo diante dos obstaculos e das dificuldades"),
    ("como desenvolver paciencia", "praticar a calma aceitar o que nao se pode controlar e focar no presente desenvolvem a paciencia"),
    ("o que e gratidao", "gratidao e reconhecer e valorizar o que se tem e as pessoas que nos rodeiam"),
    ("como melhorar a comunicacao", "ouvir com atencao falar com clareza e respeitar o outro sao bases de uma boa comunicacao"),
    ("o que e empatia", "empatia e a capacidade de se colocar no lugar do outro e compreender seus sentimentos e perspectivas"),
    ("como ter paz interior", "aceitar o que nao se pode mudar focar no presente e cultivar pensamentos positivos trazem paz interior"),
]

# ── Cotidiano ─────────────────────────────────────────────────────────────
cotidiano = [
    ("como passar o dia bem", "comecar com gratidao planejar as tarefas e reservar tempo para descanso torna o dia mais produtivo"),
    ("o que fazer quando estiver triste", "conversar com alguem de confianca praticar uma atividade que goste e descansar ajuda a melhorar o humor"),
    ("como se motivar", "lembrar dos seus objetivos celebrar pequenas conquistas e buscar inspiracao em pessoas que admira"),
    ("o que fazer nas horas vagas", "ler praticar um esporte aprender algo novo ou simplesmente descansar sao otimas opcoes"),
    ("como dormir melhor", "manter horarios regulares evitar telas antes de dormir e ter um ambiente calmo melhora o sono"),
    ("o que comer para ter energia", "frutas vegetais graos integrais e proteinas dao energia sustentada ao longo do dia"),
    ("como lidar com o estresse", "fazer exercicios respirar fundo conversar com amigos e organizar as tarefas reduzem o estresse"),
    ("como aprender algo novo", "praticar regularmente buscar boas fontes e ter paciencia consigo mesmo sao chaves para aprender"),
    ("o que fazer para se sentir bem", "cuidar do corpo da mente e das relacoes e praticar atividades que gosta traz bem estar"),
    ("como economizar dinheiro", "planejar os gastos evitar compras por impulso e guardar uma parte da renda regularmente"),
    ("como ser mais organizado", "ter um lugar para cada coisa planejar o dia e eliminar o que nao e necessario ajuda na organizacao"),
    ("o que fazer quando nao sabe o que fazer", "respirar pensar com calma conversar com alguem e dar um passo de cada vez"),
    ("como fazer amigos", "ser genuino mostrar interesse pelo outro estar disponivel e ser confiavel atrai boas amizades"),
    ("como pedir desculpas", "ser sincero assumir a responsabilidade e mostrar mudanca real e a forma certa de pedir desculpas"),
    ("como dar um bom conselho", "ouvir primeiro entender a situacao e falar com cuidado e honestidade ao dar conselhos"),
]

# ── Natureza e mundo ──────────────────────────────────────────────────────
natureza = [
    ("o que e o tempo", "o tempo e a sequencia de momentos que compoe nossa existencia e nao para por ninguem"),
    ("por que a natureza e importante", "a natureza e a base de toda vida no planeta provendo ar agua alimento e equilíbrio"),
    ("o que e o vento", "o vento e o movimento do ar causado pelas diferencas de temperatura e pressao na atmosfera"),
    ("o que causa a chuva", "a evaporacao da agua forma nuvens que quando carregadas liberam a agua em forma de chuva"),
    ("por que o ceu e azul", "a luz do sol se espalha na atmosfera e o azul e dispersado mais que as outras cores"),
    ("o que e uma floresta", "uma floresta e um ecossistema rico em arvores plantas e animais que vive em equilibrio"),
    ("por que e importante cuidar do planeta", "o planeta e nossa unica casa e cuidar dele garante a vida para as futuras geracoes"),
    ("o que e biodiversidade", "biodiversidade e a variedade de seres vivos que habitam a terra e e essencial para o equilibrio"),
    ("por que a agua e preciosa", "a agua e fundamental para toda forma de vida e sem ela nao existe existencia no planeta"),
    ("o que e o fogo", "o fogo e a reacao de combustao que libera calor e luz e pode ser util ou destrutivo"),
]

# ── Trabalho e esforco ────────────────────────────────────────────────────
trabalho = [
    ("como ter sucesso no trabalho", "ser dedicado honesto pontual e colaborativo sao qualidades que levam ao sucesso profissional"),
    ("o que faz um bom lider", "um bom lider inspira ouve sua equipe tem visao clara e age com integridade"),
    ("como lidar com criticas no trabalho", "ouvir com abertura refletir sobre o que faz sentido e agir para melhorar e a melhor resposta"),
    ("o que e trabalho em equipe", "trabalho em equipe e a soma de esforcos individuais em direcao a um objetivo comum"),
    ("como ser mais criativo", "explorar novas perspectivas questionar o obvio e permitir se errar estimulam a criatividade"),
    ("o que motiva as pessoas no trabalho", "reconhecimento proposito aprendizado e boas relacoes sao grandes motivadores profissionais"),
    ("como tomar boas decisoes", "coletar informacoes considerar as consequencias ouvir opinioes e confiar na intuicao"),
    ("o que e etica profissional", "etica profissional e agir com honestidade respeito e responsabilidade no ambiente de trabalho"),
    ("como aprender com os erros", "analisar o que deu errado entender o porque e ajustar a abordagem transforma erros em crescimento"),
    ("o que e responsabilidade", "responsabilidade e assumir as proprias acoes e suas consequencias com honestidade"),
]

# ── Relacionamentos ───────────────────────────────────────────────────────
relacionamentos = [
    ("como ter um bom relacionamento", "comunicacao honesta respeito cumplicidade e espaco individual sao bases de um bom relacionamento"),
    ("o que e respeito", "respeito e reconhecer o valor e os limites do outro e trata lo com dignidade"),
    ("como resolver conflitos", "ouvir o outro com atencao buscar entendimento e ceder quando necessario resolve conflitos"),
    ("o que torna uma amizade verdadeira", "presenca nos momentos dificeis honestidade e lealdade sao marcas de uma amizade verdadeira"),
    ("como demonstrar carinho", "estar presente ouvir ajudar e expressar afeto com palavras e acoes demonstra carinho"),
    ("o que e confianca", "confianca e a certeza de que o outro age com honestidade e tem boas intencoes"),
    ("como reconquistar a confianca", "ser consistente honesto e paciente no tempo e o caminho para reconquistar a confianca"),
    ("o que e lealdade", "lealdade e manter o compromisso com o outro mesmo nos momentos de dificuldade"),
    ("como ajudar alguem que esta sofrendo", "estar presente ouvir sem julgar e oferecer apoio pratico sao formas de ajudar quem sofre"),
    ("o que e companheirismo", "companheirismo e caminhar junto dividindo experiencias apoios e alegrias com o outro"),
]

# ── Saúde ──────────────────────────────────────────────────────────────────
saude = [
    ("como manter a saude", "exercicios regulares alimentacao equilibrada sono adequado e gestao do estresse sao fundamentais"),
    ("o que e saude mental", "saude mental e o bem estar emocional e psicologico que permite viver de forma equilibrada"),
    ("como cuidar da mente", "praticar meditacao manter relacoes saudaveis ter proposito e pedir ajuda quando necessario"),
    ("por que o exercicio e importante", "o exercicio fortalece o corpo melhora o humor aumenta a energia e previne doencas"),
    ("o que e uma alimentacao saudavel", "variedade de frutas legumes graos integrais proteinas e poucos alimentos processados"),
    ("como ter mais energia no dia a dia", "dormir bem beber agua comer bem praticar exercicios e ter momentos de lazer dao energia"),
    ("o que causa o cansaco", "falta de sono ma alimentacao estresse e falta de atividade fisica sao causas comuns de cansaco"),
    ("como reduzir a ansiedade", "respiracao profunda exercicios conversar com alguem e limitar o tempo de telas ajudam"),
    ("o que e bem estar", "bem estar e sentir se saudavel satisfeito e em equilibrio em todas as areas da vida"),
    ("como ter uma vida mais saudavel", "pequenas mudancas diarias como caminhar beber agua e dormir cedo fazem grande diferenca"),
]

# ── Templates para gerar variações ────────────────────────────────────────
temas = [
    ("amor", "um sentimento que conecta as pessoas e traz alegria e cuidado mutuo"),
    ("familia", "a base emocional que nos sustenta nos momentos bons e ruins da vida"),
    ("trabalho", "uma forma de contribuir para a sociedade e realizar os proprios sonhos"),
    ("saude", "o bem mais precioso que temos e deve ser cuidado com atencao e respeito"),
    ("amizade", "um laco de confianca e afeto que enriquece a vida de quem a cultiva"),
    ("natureza", "a fonte de toda vida e um presente que devemos proteger"),
    ("tempo", "um recurso precioso que nao volta e deve ser usado com sabedoria"),
    ("conhecimento", "uma ferramenta poderosa que liberta e abre portas para novas possibilidades"),
    ("paciencia", "uma virtude que permite esperar com calma pelo momento certo"),
    ("coragem", "a forca interior que nos move a agir mesmo diante do medo e da incerteza"),
    ("bondade", "a qualidade de tratar o outro com gentileza e generosidade sem esperar nada em troca"),
    ("persistencia", "a capacidade de continuar tentando mesmo quando as coisas ficam dificeis"),
    ("respeito", "a base de qualquer relacao saudavel e o reconhecimento do valor do outro"),
    ("honestidade", "a virtude de falar e agir com verdade mesmo quando e dificil"),
    ("gratidao", "o habito de reconhecer e valorizar o que se tem e as pessoas ao redor"),
]

question_templates = [
    ("o que e {tema}", "e {def}"),
    ("fale sobre {tema}", "{tema} e {def}"),
    ("por que {tema} e importante", "{tema} e importante porque {def}"),
    ("qual o valor de {tema}", "o valor de {tema} e imenso pois {def}"),
    ("como cultivar {tema}", "para cultivar {tema} e preciso praticar diariamente pois {def}"),
    ("me fale sobre {tema}", "posso dizer que {tema} e {def}"),
    ("o que voce pensa sobre {tema}", "penso que {tema} e {def}"),
]

def generate_template_pairs():
    pairs = []
    for tema, defi in temas:
        for q_tmpl, a_tmpl in question_templates:
            q = q_tmpl.format(tema=tema)
            a = a_tmpl.format(tema=tema, defi=defi, **{"def": defi})
            pairs.append((q, a))
    return pairs

# ── Respostas curtas para perguntas diretas ────────────────────────────────
diretas = [
    ("tudo bem", "tudo bem obrigado e voce como esta passando"),
    ("como voce esta", "estou bem pronto para conversar e ajudar no que precisar"),
    ("ola", "ola como posso ajudar voce hoje"),
    ("oi", "oi tudo bem como posso ser util"),
    ("bom dia", "bom dia que o seu dia seja cheio de alegria e disposicao"),
    ("boa tarde", "boa tarde espero que esteja tendo um dia produtivo"),
    ("boa noite", "boa noite descanse bem e tenha sonhos tranquilos"),
    ("obrigado", "de nada fico feliz em poder ajudar sempre que precisar"),
    ("valeu", "disponha espero ter sido util para voce"),
    ("ate mais", "ate mais foi um prazer conversar com voce"),
    ("tchau", "tchau cuide se e volte sempre que quiser conversar"),
    ("preciso de ajuda", "estou aqui para ajudar pode falar o que voce precisa"),
    ("nao sei o que fazer", "vamos pensar juntos pode me contar mais sobre a situacao"),
    ("estou triste", "sinto muito que esteja assim conversar pode ajudar me conte o que esta acontecendo"),
    ("estou feliz", "que otimo a felicidade e algo a ser celebrado o que trouxe tanta alegria"),
    ("estou cansado", "o descanso e essencial tente reservar um tempo so para voce recuperar as energias"),
    ("nao entendi", "pode me dizer o que ficou confuso para eu explicar melhor"),
    ("pode repetir", "claro com prazer vou explicar novamente da forma mais clara possivel"),
    ("voce sabe de tudo", "sei algumas coisas mas sempre ha mais para aprender e descobrir"),
    ("me ajuda", "claro estou aqui me diga o que precisa e farei o possivel para ajudar"),
]

# ── Montar corpus ─────────────────────────────────────────────────────────
all_pairs = []
all_pairs.extend(proverbios)
all_pairs.extend(vida)
all_pairs.extend(cotidiano)
all_pairs.extend(natureza)
all_pairs.extend(trabalho)
all_pairs.extend(relacionamentos)
all_pairs.extend(saude)
all_pairs.extend(diretas)
all_pairs.extend(generate_template_pairs())

# Shuffle para variar os padrões no corpus
random.shuffle(all_pairs)

# Repetir 3x para dar mais exemplos de treino
base = list(all_pairs)
all_pairs = base + base + base
random.shuffle(all_pairs)

output_file = "chat_corpus.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for user_text, asst_text in all_pairs:
        line = f"user {user_text} assistant {asst_text}\n"
        f.write(line)

print(f"Corpus gerado: {output_file}")
print(f"Total de pares: {len(all_pairs)}")
print(f"Exemplo:")
print(f"  user {all_pairs[0][0]} assistant {all_pairs[0][1]}")
