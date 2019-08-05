from os.path import dirname, join

ROOT = dirname(dirname(__file__))
DATA = join(ROOT, 'data')
TRAIN_DATA = join(DATA, 'sst2', 'stsa.binary.phrases.train')
VALIDATION_DATA = join(DATA, 'sst2', 'stsa.binary.dev')
TEST_DATA = join(DATA, 'sst2', 'stsa.binary.test')
PRETRAINED_EMBEDDINGS_FILE = join(DATA, 'GoogleNews-vectors-negative300.bin')
CHECKPOINT_PATH = join(ROOT, "models")