import os

from model import *
from utils import test_eval


class DRDetector:
    def __init__(self, train_config, model_config, data):
        self.model_config = model_config
        self.train_config = train_config
        self.data = data
        self.model = DR(**model_config).to(train_config['device'])
        self.model_dir = "./saved_models"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.train_dataset_names = [d.name for d in data['train']]

    def get_model_path(self, seed):
        return os.path.join(self.model_dir,
                            f"{self.model_config['model']}_seed{seed}.pth")

    def model_exists(self, seed):
        return os.path.exists(self.get_model_path(seed))

    def save_model(self, seed):
        save_data = {'state_dict': self.model.state_dict(), 'model_config': self.model_config,
                     'train_dataset_names': self.train_dataset_names,
                     'cluster_centers': {name: self.model.cluster_centers_dict[name] for name in
                                         self.train_dataset_names},
                     'cluster_mlp_keys': list(self.model.cluster_mlps.keys())}
        torch.save(save_data, self.get_model_path(seed))
        print(f"Model saved for seed {seed}")

    def load_model(self, seed):
        if not self.model_exists(seed):
            raise FileNotFoundError(f"No model found for seed {seed}")
        save_data = torch.load(self.get_model_path(seed))

        if set(save_data['train_dataset_names']) != set(self.train_dataset_names):
            raise ValueError(
                f"Training datasets mismatch. Saved: {save_data['train_dataset_names']}, Current: {self.train_dataset_names}")
        for name, centers in save_data['cluster_centers'].items():
            self.model.cluster_centers_dict[name] = centers.to(self.train_config['device'])

        for name in self.train_dataset_names:
            if name not in self.model.cluster_mlps:
                self.model.cluster_mlps[name] = nn.Linear(self.model.in_feats * self.model.num_hops,
                                                          self.model.h_feats * self.model.num_hops).to(
                    self.train_config['device'])
        self.model.load_state_dict(save_data['state_dict'])
        print(f"Model loaded for seed {seed}")

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'],
                                     weight_decay=self.model_config['weight_decay'])

        for e in range(self.model_config['epoch']):
            for didx, train_data in enumerate(self.data['train']):
                self.model.train()
                train_graph = self.data['train'][didx].graph.to(self.train_config['device'])
                if e == 0:
                    labels = train_graph.ano_labels
                    normal_idx = (labels == 0)
                    self.model.compute_residual_prototypes(train_graph, normal_idx, self.data['train'][didx].name)
                loss = self.model.computer_loss(train_graph, self.data['train'][didx].name)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print('Finish Training for {} epochs!'.format(self.model_config['epoch']))

    def test(self):
        test_score_list = {}
        self.model.eval()
        for didx, test_data in enumerate(self.data['test']):
            test_graph = test_data.graph.to(self.train_config['device'])
            labels = test_graph.ano_labels
            query_scores = self.model.get_anomaly_score(test_graph)
            test_score = test_eval(labels, query_scores)
            test_data_name = self.train_config['testdsets'][didx]
            test_score_list[test_data_name] = {'AUROC': test_score['AUROC'], 'AUPRC': test_score['AUPRC'], }
        return test_score_list
