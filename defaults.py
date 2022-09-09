import re
import argparse


def match_model_name(model_name):
    """
    Asserts that the model name is valid and extract the values.
    """
    # Model has to match the following pattern:
    # XXXXX_YY_ZZ where XXXXX is the model name.
    # XXXX can have any length, but YY and ZZ have to be av, a or v.
    regex = re.compile(r'^([a-zA-Z0-9]+)_([a|v]{1,2})_([a|v]{1,2})$')
    match = regex.match(model_name)
    if match is None:
        raise argparse.ArgumentTypeError(
            'Model name must be in the following format: '
            'XXXXX_YY_ZZ where XXXX is the model name and YY and ZZ '
            'have to be av, a or v.')
    return match.groups()


class Argparser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('Loading argparser...')

        # Model config
        self.add_argument('-m', '--model', required=True, type=str, help='Model to use for training')
        self.add_argument('--ckpt_path', type=str, default=None, help='Resume from checkpoint')
        self.add_argument('--find_lr', action='store_true', default=False, help='Find learning rate')
        self._add_model_cfg()
        self.add_argument('--batch_size', type=int, default=10, help='Batch size')

    def add_train_cfg(self):
        parser = self.add_argument_group("train_cfg")

        parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
        parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
        parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
        parser.add_argument('--patience', type=int, default=10, help='Patience')

        return self

    def add_test_cfg(self):
        parser = self.add_argument_group("test_cfg")
        parser.add_argument('--test_dir', type=str, required=True, default=None, help='Test evaluation directory')

        return self

    def build(self):
        args = self.parse_args()
        print('Argparser loaded!')
        print('Adding atributes to argparser...')
        return args
