import os, sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'generation', 'data')
PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'img')

# import oe_eval
oe_eval_dir = os.path.abspath(os.path.join(ROOT_DIR, 'oe-eval', 'oe-eval-internal'))
try:
    sys.path.append(oe_eval_dir)
    import oe_eval
except ImportError:
    print(f"Warning: Unable to import oe_eval from {oe_eval_dir}")