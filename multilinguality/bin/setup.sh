mkdir -p out

git clone https://github.com/SapienzaNLP/mwsd-datasets.git

pushd mwsd-datasets/data/
tar -xvf multilingual_wsd_wn_v1.0.tar.gz

cd ../inventory_building

wget -c https://babelnet.org/data/4.0/BabelNet-API-4.0.1.zip
unzip BabelNet-API-4.0.1.zip
cp -r BabelNet-API-4.0.1/resources .
cp -r BabelNet-API-4.0.1/config .
cp ../../res/jlt.var.properties mwsd-datasets/inventory_building/config/

wget -c http://wordnetcode.princeton.edu/3.0/WordNet-3.0.tar.gz
tar -xvf WordNet-3.0.tar.gz

cp ../../config/* config/
