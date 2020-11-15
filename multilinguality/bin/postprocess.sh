OUT=$1

bash bin/clean.sh inventories/it/inventory.it.withgold.sorted.txt ${OUT}/dictionaries it
bash bin/clean.sh inventories/fr/inventory.fr.withgold.sorted.txt ${OUT}/dictionaries fr
bash bin/clean.sh inventories/de/inventory.de.withgold.sorted.txt ${OUT}/dictionaries de
bash bin/clean.sh inventories/es/inventory.es.withgold.sorted.txt ${OUT}/dictionaries es
