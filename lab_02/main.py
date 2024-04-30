from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    PER,
    NamesExtractor,
    Doc
)

from yargy import rule, Parser, and_, or_, not_
from yargy.interpretation import fact
from yargy.predicates import gram, dictionary, gte, lte, type
from yargy.relations import gnc_relation
from yargy.tokenizer import INT

emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)

segmenter = Segmenter()

morph_vocab = MorphVocab()
extractor = NamesExtractor(morph_vocab)

with open('wiki.txt', encoding='utf-8') as f:
    text = f.read()

doc = Doc(text)

doc.segment(segmenter)
doc.tag_morph(morph_tagger)
doc.tag_ner(ner_tagger)

for token in doc.tokens:
    token.lemmatize(morph_vocab)

for span in doc.spans:
    span.normalize(morph_vocab)

for span in doc.spans:
    if span.type == PER:
        span.extract_fact(extractor)

for rec in {_.normal: _.fact.as_dict for _ in doc.spans if _.fact is not None}.items():
    print(rec)

print("====================")

for span in doc.spans:
    print(span)

print("====================")

LangWikiFact = fact(
    'Wiki_Language_Partition',
    ['language', 'type']
)

LANG = dictionary({'русский', 'русскоязычный', 'английский', 'англоязычный', 'немецкий', 'польский', 'испанский'})
TYPE = dictionary({'википедия', 'раздел', 'секция'})

LangWikiRule = rule(
    LANG.interpretation(LangWikiFact.language.inflected()),
    TYPE.interpretation(LangWikiFact.type.inflected())
).interpretation(LangWikiFact)

parser = Parser(LangWikiRule)
for match in parser.findall(text):
    print(match.fact)

print("====================")

Date = fact(
    'Date',
    ['day', 'month', 'year']
)

# Месяц названием
MONTHS_NAMES = dictionary({
    'январь',
    'февраль',
    'март',
    'апрель',
    'май',
    'июнь',
    'июль',
    'август',
    'сентябрь',
    'октябрь',
    'ноябрь',
    'декабрь'
}).interpretation(Date.month)

# День - число от 1 до 31
DAY = and_(
    gte(1),
    lte(31)
).interpretation(Date.day)

# Месяц числом - от 1 до 12
MONTH_NUM = and_(
    gte(1),
    lte(12)
).interpretation(Date.month)

# Год - любое положительное число
YEAR = and_(
    type(INT),
    gte(1)
).interpretation(Date.year)

# Дата в одном из форматов: 01.01.2001, 01-01-2001, 01 января 2001, 2001 год
DATE = or_(
    rule(DAY.optional(), MONTHS_NAMES, rule(YEAR, dictionary({'год'}).optional()).optional()),
    rule(DAY, '-', MONTH_NUM, '-', YEAR),
    rule(DAY, '.', MONTH_NUM, '.', YEAR),
    rule(YEAR, dictionary({'год'}))
).interpretation(Date)

parser = Parser(DATE)
for match in parser.findall(text):
    print(match.fact)

print("====================")

NumberOfObjects = fact(
    "Numer_of_objects",
    ['amount', 'multiplier', 'object']
)

INF = dictionary({'статья', 'диск', 'изображение', 'страница'})
CURRENCY = dictionary({'рубль', 'доллар', 'евро', 'злотый', '€'})
MUL = dictionary({'десяток', 'дюжина', 'сотня', 'тысяча', 'тыс', 'миллион'})

num = fact('num', ['n'])

INTEGER = or_(
    rule(type(INT))
)
POINT = dictionary({'.', ','})
FLOAT = rule(
    INTEGER,
    POINT,
    type(INT)
)

NUMBER = or_(
    rule(INTEGER),
    rule(FLOAT)
).interpretation(NumberOfObjects.amount)

NUMBER_OF_OBJECTS = or_(
    rule(
        NUMBER,
        MUL.interpretation(NumberOfObjects.multiplier).optional(),
        INF.interpretation(NumberOfObjects.object)
    ),
    rule(
        NUMBER,
        MUL.interpretation(NumberOfObjects.multiplier).optional(),
        CURRENCY.interpretation(NumberOfObjects.object)
    )
).interpretation(NumberOfObjects)

parser = Parser(NUMBER_OF_OBJECTS)
for match in parser.findall(text):
    print(match.fact)

print("====================")
