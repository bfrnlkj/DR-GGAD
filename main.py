import argparse
import logging
import warnings
from datetime import datetime

import pandas as pd

from train_test import DRDetector
from utils import *


def calculate_mean(data_list):
    if not data_list:
        return 0.0
    return sum(data_list) / len(data_list)


def calculate_std(data_list):
    if len(data_list) < 2:
        return 0.0
    mean = calculate_mean(data_list)
    variance = sum((x - mean) ** 2 for x in data_list) / (len(data_list) - 1)
    return variance ** 0.5


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def setup_logging():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = "./log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    logging.info("Logging is set up.")


warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='DR')
args = parser.parse_known_args()[0]

alpha_range = [0.1]

datasets_test = ['cora', 'citeseer', 'ACM', 'BlogCatalog', 'Facebook', 'weibo', 'Reddit', 'Amazon']
datasets_train = ['pubmed', 'Flickr', 'questions', 'YelpChi']

model = args.model
model_result = {'name': model}
setup_logging()

logging.info(f'Training on {len(datasets_train)} datasets: {datasets_train}')
logging.info(f'Test on {len(datasets_test)} datasets: {datasets_test}')

train_config = {'device': 'cuda:0', 'testdsets': datasets_test}
dims = 64
data_train = [Dataset(dims, name) for name in datasets_train]
data_test = [Dataset(dims, name) for name in datasets_test]  # CPU

all_results = []

for alpha in alpha_range:
    logging.info(f"Running Grid Search: alpha={alpha}")

    model_config = {"model": "ARC", "lr": 1e-5, "drop_rate": 0.2, "h_feats": 1024, "num_prompt": 10, "num_hops": 2,
                    "weight_decay": 5e-5, "in_feats": 64, "num_layers": 4, "activation": "ELU", "alpha": alpha,
                     "k": 1, "epoch": 60}

    logging.info(model_config)
    for tr_data in data_train:
        tr_data.propagated(model_config['num_hops'])

    for te_data in data_test:
        te_data.propagated(model_config['num_hops'])

    auc_dict = {name: [] for name in datasets_test}
    pre_dict = {name: [] for name in datasets_test}
    for t in [5]:
        seed = t
        set_seed(seed)
        train_config['seed'] = seed

        data = {'train': data_train, 'test': data_test}
        detector = DRDetector(train_config, model_config, data)

        try:
            if detector.model_exists(seed):
                logging.info(f"Model for seed {seed} exists. Attempting to load...")
                detector.load_model(seed)
                logging.info("Model loaded successfully")
            else:
                logging.info(f"No model found for seed {seed}. Starting training...")
                detector.train()
                detector.save_model(seed)
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            logging.info("Reinitializing and training new model...")
            detector = DRDetector(train_config, model_config, data)
            detector.train()
            detector.save_model(seed)

        test_score_list = detector.test()

        for test_data_name, test_score in test_score_list.items():
            auc_dict[test_data_name].append(test_score['AUROC'])
            pre_dict[test_data_name].append(test_score['AUPRC'])
            logging.info(f"Test on {test_data_name}:")
            logging.info(f"  AUROC: {test_score['AUROC']:.4f}, AUPRC: {test_score['AUPRC']:.4f}")

    for test_data_name in datasets_test:
        auc_mean = calculate_mean(auc_dict[test_data_name])
        auc_std = calculate_std(auc_dict[test_data_name])
        pre_mean = calculate_mean(pre_dict[test_data_name])
        pre_std = calculate_std(pre_dict[test_data_name])

        all_results.append({'alpha': alpha, 'Dataset': test_data_name,
                            'AUROC (Mean ± Std)': f'{auc_mean:.4f} ± {auc_std:.4f}',
                            'AUPRC (Mean ± Std)': f'{pre_mean:.4f} ± {pre_std:.4f}', 'AUROC_Mean': auc_mean,
                            'AUPRC_Mean': pre_mean})

results_df = pd.DataFrame(all_results)
