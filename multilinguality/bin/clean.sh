INPUT=$1
OUT=$2
LANG=$3

LEMMAPOS2OFF=${OUT}/lemma_pos2offsets.${LANG}.txt
LEMMAPOS=${OUT}/lemma_pos.${LANG}.txt

sed "s/#ADV/#r/" $INPUT | sed "s/#ADJ/#a/" | sed "s/#NOUN/#n/" | sed "s/#VERB/#v/" > $LEMMAPOS2OFF
cut -f1 $LEMMAPOS2OFF | awk '{print $1 " 1"}' > $LEMMAPOS
