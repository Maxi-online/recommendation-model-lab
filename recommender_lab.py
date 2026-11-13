"""Portable implicit-feedback benchmark; independent of source accounts and SDKs."""
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import platform
import numpy as np


@dataclass(frozen=True)
class Interaction:
    user: str
    item: str
    timestamp: int


def load_csv(path):
    rows = []
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'user_id', 'item_id', 'timestamp'} <= set(reader.fieldnames or []):
            raise ValueError('CSV requires user_id, item_id and integer timestamp columns')
        for number, row in enumerate(reader, 2):
            if not row['user_id'].strip() or not row['item_id'].strip():
                raise ValueError(f'Missing identifier on CSV line {number}')
            rows.append(Interaction(row['user_id'].strip(), row['item_id'].strip(), int(row['timestamp'])))
    return rows


def temporal_split(rows, train_fraction=2 / 3, validation_fraction=1 / 6):
    """Global timestamp boundaries are selected before fitting any model."""
    if len(rows) < 6 or not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError('Need at least six events and positive train/validation/test fractions')
    ordered = sorted(rows, key=lambda x: (x.timestamp, x.user, x.item))
    train_cut = ordered[max(0, int(len(ordered) * train_fraction) - 1)].timestamp
    validation_cut = ordered[max(0, int(len(ordered) * (train_fraction + validation_fraction)) - 1)].timestamp
    train = [r for r in ordered if r.timestamp <= train_cut]
    validation = [r for r in ordered if train_cut < r.timestamp <= validation_cut]
    test = [r for r in ordered if r.timestamp > validation_cut]
    if not train or not validation or not test:
        raise ValueError('Timestamp ties leave an empty split; supply more distinct event times')
    return train, validation, test


def synthetic_interactions(seed, users=80, groups=4, items_per_group=30, events_per_user=18):
    """Generated preference groups; user IDs and item IDs identify no real person."""
    rng = np.random.default_rng(seed)
    sequences = [rng.choice(items_per_group, events_per_user, replace=False) for _ in range(users)]
    return [Interaction(f'demo-user-{u:03}', f'item-{u % groups:02}-{int(sequences[u][t]):03}', t * users + u)
            for t in range(events_per_user) for u in range(users)]


class Popularity:
    def fit(self, rows):
        self.counts = Counter(r.item for r in rows)
        self.items = sorted(self.counts)
        self.seen = defaultdict(set)
        for row in rows:self.seen[row.user].add(row.item)
        return self

    def recommend(self, user, k=10):
        return sorted((i for i in self.items if i not in self.seen.get(user, set())),
                      key=lambda i: (-self.counts[i], i))[:k]


class BPRMatrixFactorization:
    """SGD-trained user/item embeddings with a pairwise logistic objective."""
    def __init__(self, dimensions=16, epochs=30, learning_rate=0.05, regularization=0.002, seed=19):
        if min(dimensions, epochs) < 1 or learning_rate <= 0 or regularization < 0:
            raise ValueError('Invalid optimizer configuration')
        self.dimensions, self.epochs = dimensions, epochs
        self.learning_rate, self.regularization, self.seed = learning_rate, regularization, seed

    def fit(self, rows):
        if not rows:raise ValueError('Training interactions cannot be empty')
        self.users = sorted({r.user for r in rows})
        self.items = sorted({r.item for r in rows})
        self.user_index = {value: i for i, value in enumerate(self.users)}
        self.item_index = {value: i for i, value in enumerate(self.items)}
        self.seen = defaultdict(set)
        for row in rows:self.seen[row.user].add(row.item)
        self.fallback = Popularity().fit(rows)
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.1, (len(self.users), self.dimensions))
        self.item_factors = rng.normal(0, 0.1, (len(self.items), self.dimensions))
        # Only training positives enter negative sampling. Held-out labels are
        # deliberately unavailable here, including potential future positives.
        negatives = {u: np.array([j for j, item in enumerate(self.items) if item not in self.seen[u]], dtype=np.int64) for u in self.users}
        pairs = sorted({(self.user_index[r.user], self.item_index[r.item]) for r in rows})
        for _ in range(self.epochs):
            for index in rng.permutation(len(pairs)):
                u, positive = pairs[int(index)]
                candidates = negatives[self.users[u]]
                if not len(candidates):continue
                negative = int(rng.choice(candidates))
                p = self.user_factors[u].copy()
                qi, qj = self.item_factors[positive].copy(), self.item_factors[negative].copy()
                margin = float(p @ (qi - qj))
                gradient = 1 / (1 + math.exp(float(np.clip(margin, -40, 40))))
                rate, reg = self.learning_rate, self.regularization
                self.user_factors[u] += rate * (gradient * (qi - qj) - reg * p)
                self.item_factors[positive] += rate * (gradient * p - reg * qi)
                self.item_factors[negative] += rate * (-gradient * p - reg * qj)
        return self

    def recommend(self, user, k=10):
        if k < 1:raise ValueError('k must be positive')
        if user not in self.user_index:return self.fallback.recommend(user, k)
        scores = self.item_factors @ self.user_factors[self.user_index[user]]
        ranked = sorted(range(len(self.items)), key=lambda i: (-float(scores[i]), self.items[i]))
        return [self.items[i] for i in ranked if self.items[i] not in self.seen[user]][:k]

    def save(self, path):
        metadata = {'users': self.users, 'items': self.items, 'seen': {u: sorted(v) for u, v in self.seen.items()},
                    'counts': dict(self.fallback.counts), 'dimensions': self.dimensions, 'seed': self.seed}
        np.savez(path, users=self.user_factors, items=self.item_factors, metadata=np.array(json.dumps(metadata)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata']))
            model = cls(dimensions=meta['dimensions'], seed=meta['seed'])
            model.user_factors, model.item_factors = data['users'].copy(), data['items'].copy()
        model.users, model.items = meta['users'], meta['items']
        if model.user_factors.shape != (len(model.users), model.dimensions) or model.item_factors.shape != (len(model.items), model.dimensions):
            raise ValueError('Checkpoint dimensions do not match metadata')
        model.user_index = {u: i for i, u in enumerate(model.users)}
        model.item_index = {i: n for n, i in enumerate(model.items)}
        model.seen = defaultdict(set, {u: set(v) for u, v in meta['seen'].items()})
        model.fallback = Popularity()
        model.fallback.items, model.fallback.seen = model.items, model.seen
        model.fallback.counts = Counter(meta['counts'])
        return model


def evaluate(model, held_out, k=10):
    targets = defaultdict(set)
    catalog = set(model.items)
    cold_user_events = cold_item_events = seen_pair_events = 0
    for row in held_out:
        if row.user not in model.seen:
            cold_user_events += 1
            continue
        if row.item not in catalog:
            cold_item_events += 1
            continue
        if row.item in model.seen[row.user]:
            seen_pair_events += 1
            continue
        targets[row.user].add(row.item)
    recalls, ndcgs = [], []
    for user, relevant in sorted(targets.items()):
        recommendations = model.recommend(user, k)
        hits = [float(item in relevant) for item in recommendations]
        recalls.append(sum(hits) / len(relevant))
        dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
        ndcgs.append(dcg / ideal)
    return {'recall_at_k': float(np.mean(recalls)) if recalls else None,
            'ndcg_at_k': float(np.mean(ndcgs)) if ndcgs else None,
            'evaluated_users': len(targets), 'input_events': len(held_out),
            'cold_user_events_excluded': cold_user_events, 'cold_item_events_excluded': cold_item_events,
            'previously_seen_pairs_excluded': seen_pair_events, 'k': k}


def benchmark(seed=19, rows=None, epochs=30):
    train, validation, test = temporal_split(rows if rows is not None else synthetic_interactions(seed))
    popularity = Popularity().fit(train)
    learned = BPRMatrixFactorization(seed=seed, epochs=epochs).fit(train)
    return {'seed': seed, 'split': {'train_events': len(train), 'validation_events': len(validation), 'test_events': len(test),
                                  'train_max_time': max(x.timestamp for x in train), 'test_min_time': min(x.timestamp for x in test)},
            'protocol': 'Global chronological split before fit; train-only vocabulary and negatives; validation not used for tuning; test ranking excludes training positives',
            'popularity': evaluate(popularity, test), 'bpr_matrix_factorization': evaluate(learned, test)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 19, 43])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rows = load_csv(args.csv) if args.csv else None
    report = {'dataset': 'operator-supplied CSV' if rows is not None else 'synthetic preference groups',
              'scope': 'Portable baseline experiment, not a production or seven-adapter quality benchmark',
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'device': 'cpu'},
              'runs': [benchmark(seed, rows, args.epochs) for seed in args.seeds]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':main()
