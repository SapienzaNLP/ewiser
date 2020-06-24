import enum

from nltk.corpus import wordnet
from nltk.corpus.reader import WordNetError

def _unchanged(string: str) -> str:
    return string


def _longest(string: str) -> str:
    return max(string.replace("_", " ").split(" "), key=len)


def _first(string: str) -> str:
    return string.replace("_", " ").split(" ")[0]


def _last(string: str) -> str:
    return string.replace("_", " ").split(" ")[-1]

patching_data = {
    'ddc%1:06:01::': 'dideoxycytosine.n.01.DDC',
    'ddi%1:06:01::': 'dideoxyinosine.n.01.DDI',
    'earth%1:15:01::': 'earth.n.04.earth',
    'earth%1:17:02::': 'earth.n.01.earth',
    'moon%1:17:03::': 'moon.n.01.moon',
    'sun%1:17:02::': 'sun.n.01.Sun',
    'kb%1:23:01::': 'kilobyte.n.02.kB',
    'kb%1:23:03::': 'kilobyte.n.01.kB',
}

def patched_lemma_from_key(key, wordnet=wordnet):
    try:
        lemma = wordnet.lemma_from_key(key)
    except WordNetError as e:
        if key in patching_data:
            lemma = wordnet.lemma(patching_data[key])
        elif '%3' in key:
            lemma = wordnet.lemma_from_key(key.replace('%3', '%5'))
        else:
            raise e
    return lemma

def remove_duplicate_senses(senses):
      new_senses = []
      seen = set()
      for sense in senses:
          if sense.key() in seen:
              pass
          else:
              seen.add(sense.key())
              new_senses.append(sense)
      return new_senses

def make_offset(synset):
    return "wn:" + str(synset.offset()).zfill(8) + synset.pos()


class MWEStrategy(enum.Enum):

    UNCHANGED = _unchanged
    LONGEST = _longest
    FIRST = _first
    LAST = _last