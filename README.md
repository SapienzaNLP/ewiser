# EWISER (Enhanced WSD Integrating Synset Embeddings and Relations)
This repo hosts the code necessary to reproduce the results of our ACL 2020 paper, *Breaking Through the 80% Glass Ceiling: Raising the State of the Art in Word Sense Disambiguation by Incorporating Knowledge Graph Information*, by Michele Bevilacqua and Roberto Navigli, which you can read on [ACL Anthology](https://www.aclweb.org/anthology/2020.acl-main.255/).

You will also find a simple [spacy](https://spacy.io/) plugin that makes it easy to use EWISER in your own project!

EWISER relies on the [`fairseq`](https://github.com/pytorch/fairseq) library.

## NEWS:
Check out the [Multilingual](#Multilingual) section below!

## How to Cite
```
@inproceedings{bevilacqua-navigli-2020-breaking,
    title = "Breaking Through the 80{\%} Glass Ceiling: {R}aising the State of the Art in Word Sense Disambiguation by Incorporating Knowledge Graph Information",
    author = "Bevilacqua, Michele  and Navigli, Roberto",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.255",
    pages = "2854--2864",
    abstract = "Neural architectures are the current state of the art in Word Sense Disambiguation (WSD). However, they make limited use of the vast amount of relational information encoded in Lexical Knowledge Bases (LKB). We present Enhanced WSD Integrating Synset Embeddings and Relations (EWISER), a neural supervised architecture that is able to tap into this wealth of knowledge by embedding information from the LKB graph within the neural architecture, and to exploit pretrained synset embeddings, enabling the network to predict synsets that are not in the training set. As a result, we set a new state of the art on almost all the evaluation settings considered, also breaking through, for the first time, the 80{\%} ceiling on the concatenation of all the standard all-words English WSD evaluation benchmarks. On multilingual all-words WSD, we report state-of-the-art results by training on nothing but English.",
}
```

## Installation
It is recommended to create a fresh `conda` env to use `ewiser` (e.g. `conda create -n ewiser python=3.7 pip; conda activate ewiser`).

You'll also need [`pytorch`](https://pytorch.org/get-started/locally/) 1.6, and [`torch_sparse`](https://github.com/rusty1s/pytorch_sparse). Assuming you use CUDA 10.1:
```
conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch
pip install torch-scatter torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
```

Clone this repo, install the other dependencies, and then `ewiser` as well.
```
git clone https://github.com/SapienzaNLP/ewiser.git
cd ewiser
pip install -r requirements.txt
pip install -e .
```
Now you are ready to start!

## Externally downloadable resources

EWISER English checkpoints:
* [SemCor](https://drive.google.com/file/d/1TIwCn-0NA3yUXG5FOkPgFcoP3aHJmiSZ/view?usp=sharing)
* [SemCor + untagged glosses](https://drive.google.com/file/d/1tW4PjTgdRbVvq9CGq-0ePCsgtkXnEGsN/view?usp=sharing)
* [SemCor + tagged glosses + WordNet Examples](https://drive.google.com/file/d/11RyHBu4PwS3U2wOk-Le9Ziu8R3Hc0NXV/view?usp=sharing)

EWISER multilingual checkpoints:
* [SemCor + tagged glosses + WordNet Examples](https://drive.google.com/file/d/1uYYs3izocOWx6q0yGfVku5oWNUF2-BM7/view?usp=sharing)

Datasets:
* [WSD Evaluation Framework](http://lcl.uniroma1.it/wsdeval): contains the SemCor training corpus, along with the evaluation datasets from Senseval and SemEval.
* [Multilingual Evaluation Datasets](https://github.com/SapienzaNLP/mwsd-datasets): the repo contains the French, German, Italian and Spanish datasets from SemEval 2013 and 2015. 
* The other datasets used are in `res/corpora/*/orig`.

Pre-preprocessed [SensEmBERT](http://sensembert.org/) + [LMMS](https://github.com/danlou/LMMS) embeddings (needed to train your own EWISER model):
* [Embeddings](https://drive.google.com/file/d/11v4FUMyHdpFBrkRJt8cGyy6xkM9a_Emp/view?usp=sharing)

## Multilinguality
EWISER supports all the languages for which you are able to create a mapping starting from BabelNet indices `4.0.1`.
### French/German/Italian/Spanish
1. Download the [BabelNet indices](http://nlp.uniroma1.it/resources/) (ver. 4.0.1);
2. `cd multilinguality`;
3. Set your BabelNet indices path in `multilinguality/config/babelnet.var.properties`;
4. `bash enable.sh`.
The mapping is limited to the Princeton WordNet subgraph (so you need to use the `wn` split if you plan to evaluate on [`mwsd-datasets`](https://github.com/SapienzaNLP/mwsd-datasets)).
### Other languages
Please download the multilingual mapper from [Google Drive](https://drive.google.com/file/d/1vkHLGGRbEXkmOd5QLTodDmUegipDcJyP/view?usp=sharing) and find the instructions contained there.

## Evaluate
Evaluation is run using `bin/eval_wsd.py`: 
```
# Download the WSD framework
# wget -c http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip -P res
# unzip
# WSD_FRAMEWORK=res/WSD_Evaluation_Framework

python bin/eval_wsd --checkpoints <your_checkpoint.pt> --xmls ${WSD_FRAMEWORK}/Evaluation_Datasets/ALL/ALL.data.xml ${WSD_FRAMEWORK}/Evaluation_Datasets/semeval2007/semeval2007.data.xml
```

## Spacy plugin
EWISER can be used as a `spacy` plugin. Please check `bin/annotate.py`.

## Train Your Model

### Experiment Folder and Preprocessing
To train a model from scratch, you need to set up an experiment folder containing:
* the `dict.txt` file from `res/dictionaries/dict.txt`
* the preprocessed training corpora with name `train`, `train1`, `train2` etc.
* the preprocessed validation dataset with name `valid`.

We have included our experiment directories in `res/experiments/`.

Should you need to preprocess your own corpus, you can use `bin/preprocess_wsd.py` (check out `python bin/preprocess_wsd.py --help`)!

### Training
To launch a training run, execute: 
```
cd bin
bash bin/train-ewiser.sh
```

This will train EWISER on SemCor + tagged glosses + WordNet Examples. It assumes you have downloaded the LMMS+SensEmBERT embeddings and put them in `res/embeddings/`.

You can modify hyperparameters or change the training corpora by modifyng `train-ewiser.py`. Arguments are documented in `ewiser/fairseq_ext/models/sequence_tagging.py`.

### Training Resources
#### Sense Embeddings
If you want to use your own sense embeddings in EWISER, you have to preprocess them as follows:
```shell script
python bin/get_centroids.py ${EMBEDDINGS} ${EMBEDDINGS}.centroids.txt bin/sensekeys2offsets.txt
python bin/reduce_dims.py ${EMBEDDINGS}.centroids.txt ${EMBEDDINGS}.centroids.svd512.txt -d 512
```
The sense embeddings will have to be in Glove .txt format, without a header row, and with a WN 3.0 sensekey as identifiers.

#### Edges
The adjacency matrix A in EWISER is stored as an edgelist. Each line is an edge, with three `\t`-separated values. Check `res/edges/` for examples.

## License
This project is released under the CC-BY-NC 4.0 license (see `LICENSE.txt`). If you use EWISER, please put a link to this repo.

## Acknowledgements
The authors gratefully acknowledge the support of the <a href="http://mousse-project.org">ERC Consolidator Grant MOUSSE</a> No. 726487 under the European
Union's Horizon 2020 research and innovation programme.

This work was supported in part by the MIUR under the grant "Dipartimenti di eccellenza 2018-2022" of the Department of Computer Science of the Sapienza University of Rome.
