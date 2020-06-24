import argparse
import os

from fairseq.data import Dictionary

from ewiser.fairseq_ext.data.dictionaries import MFSManager, ResourceManager, DEFAULT_DICTIONARY
from ewiser.fairseq_ext.data.wsd_dataset import WSDDataset, WSDDatasetBuilder

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=
        'Preprocessing script transforming input XMLs in Raganato format to that required by EWISER\'s training script.')
    parser.add_argument(
        '-o', '--output', required=True, help=
        'Serialized output corpus.')
    parser.add_argument(
        '-x', '--xmls', required=False, type=str, nargs='*', help=
        'Raganato XML(s), <name>.gold.key.txt should be in the same folder.')
    parser.add_argument(
        '-l', '--max-length', required=False, type=int, default=100, help=
        'Split input sequences in chunks of at most l=arg tokens.')
    parser.add_argument(
        '-i', '--input-keys', type=str, default='sensekeys', choices=['sensekeys', 'offsets', 'bnids'], help=
        """Kind of inputs keys in the <name>.gold.key.txt file. Will be all converted to WordNet offsets."""
    )
    parser.add_argument(
        '-L', '--lang', type=str, default='en', help=
        'Language of input corpora.')
    parser.add_argument('--read-by', default='text', choices=['text', 'sentence'])
    parser.add_argument(
        '--on-error', default='skip', choices=('skip', 'keep', 'raise'),
        help='What to do when some inconsistency is encountered.'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Do not print to stderr when some inconsistency is encountered.'
    )
    args = parser.parse_args()

    dictionary = Dictionary.load(DEFAULT_DICTIONARY)
    output_dictionary = ResourceManager.get_senses_dictionary(use_synsets=True)

    output = WSDDatasetBuilder(
        args.output,
        dictionary=dictionary,
        use_synsets=True,
        keep_string_data=True,
        lang=args.lang)

    for xml_path in args.xmls:
        output.add_raganato(
            xml_path=xml_path,
            max_length=args.max_length,
            input_keys=args.input_keys,
            on_error=args.on_error,
            quiet=args.quiet,
            read_by=args.read_by,
        )
    output.finalize()

    dataset = WSDDataset(args.output, dictionary=dictionary)
