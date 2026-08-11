"""Growing Neural Gas (Fritzke, 1995).

A self-organising graph that learns the topology of an input
distribution. Nodes are prototype vectors in feature space; edges are
grown by Competitive Hebbian Learning and approximate the induced
Delaunay triangulation of the data. The node count grows adaptively
until a budget is hit.

Here the input vectors are per-pixel multi-spectral signatures, so the
learned graph partitions the image into spectral land-cover clusters
without any labels or a fixed cluster count.
"""

import numpy as np


class GrowingNeuralGas:
    def __init__(
        self,
        max_nodes=50,
        max_age=50,
        lambda_step=100,
        eps_b=0.05,
        eps_n=0.006,
        alpha=0.5,
        beta=0.0005,
        rng=None,
    ):
        self.max_nodes = max_nodes
        self.max_age = max_age
        self.lambda_step = lambda_step
        self.eps_b = eps_b          # winner learning rate
        self.eps_n = eps_n          # neighbour learning rate
        self.alpha = alpha          # error decay on node insertion
        self.beta = beta            # global error decay per step
        self.rng = rng or np.random.default_rng(0)

        self.weights = None         # (n_nodes, n_dims)
        self.error = None           # (n_nodes,)
        self.edges = {}             # frozenset({i, j}) -> age

    # -- helpers ---------------------------------------------------------
    def _neighbours(self, i):
        return [next(iter(e - {i})) for e in self.edges if i in e]

    def _two_nearest(self, x):
        d = np.sum((self.weights - x) ** 2, axis=1)
        s1 = int(np.argmin(d))
        d[s1] = np.inf
        s2 = int(np.argmin(d))
        return s1, s2

    # -- training --------------------------------------------------------
    def fit(self, data, n_steps=20000, verbose=False):
        n, dim = data.shape
        # init: two random samples
        idx = self.rng.choice(n, size=2, replace=False)
        self.weights = data[idx].astype("float64").copy()
        self.error = np.zeros(2)
        self.edges = {frozenset({0, 1}): 0}

        for step in range(1, n_steps + 1):
            x = data[self.rng.integers(n)]
            s1, s2 = self._two_nearest(x)

            # accumulate error for the winner
            self.error[s1] += float(np.sum((self.weights[s1] - x) ** 2))

            # move winner and its topological neighbours
            self.weights[s1] += self.eps_b * (x - self.weights[s1])
            for nb in self._neighbours(s1):
                self.weights[nb] += self.eps_n * (x - self.weights[nb])

            # age edges from s1; refresh/create the s1-s2 edge
            for e in list(self.edges):
                if s1 in e:
                    self.edges[e] += 1
            self.edges[frozenset({s1, s2})] = 0

            # remove too-old edges and any node left isolated
            self._prune()

            # periodically insert a node where error is largest
            if step % self.lambda_step == 0 and len(self.weights) < self.max_nodes:
                self._insert_node()

            self.error *= (1.0 - self.beta)

            if verbose and step % 5000 == 0:
                print(f"  step {step:>6}  nodes={len(self.weights)}  "
                      f"edges={len(self.edges)}")

        return self

    def _prune(self):
        for e in [e for e, a in self.edges.items() if a > self.max_age]:
            del self.edges[e]
        # drop nodes with no edges
        connected = set().union(*self.edges) if self.edges else set()
        if len(connected) < len(self.weights):
            keep = sorted(connected)
            remap = {old: new for new, old in enumerate(keep)}
            self.weights = self.weights[keep]
            self.error = self.error[keep]
            self.edges = {
                frozenset({remap[a], remap[b]}): age
                for e, age in self.edges.items()
                for a, b in [tuple(e)]
            }

    def _insert_node(self):
        q = int(np.argmax(self.error))                 # highest-error node
        nbs = self._neighbours(q)
        if not nbs:
            return
        f = max(nbs, key=lambda i: self.error[i])      # its worst neighbour
        w_new = 0.5 * (self.weights[q] + self.weights[f])

        r = len(self.weights)
        self.weights = np.vstack([self.weights, w_new])
        self.error[q] *= self.alpha
        err_f = self.error[f] * self.alpha
        self.error = np.append(self.error, self.error[q])
        self.error[f] = err_f

        self.edges.pop(frozenset({q, f}), None)
        self.edges[frozenset({q, r})] = 0
        self.edges[frozenset({r, f})] = 0

    def prune_long_edges(self, factor=1.5):
        """Cut edges longer than mean + factor*std of edge length.

        A long edge bridges two spectrally distant prototypes — i.e. a
        boundary between land-cover manifolds. Removing them makes the
        connected components correspond to actual clusters, while the
        short-edge topology inside each cluster is preserved.
        """
        if not self.edges:
            return self
        lengths = {
            e: float(np.linalg.norm(self.weights[a] - self.weights[b]))
            for e in self.edges
            for a, b in [tuple(e)]
        }
        vals = np.array(list(lengths.values()))
        cutoff = vals.mean() + factor * vals.std()
        self.edges = {e: age for e, age in self.edges.items()
                      if lengths[e] <= cutoff}
        return self

    # -- inference -------------------------------------------------------
    def predict(self, data):
        """Assign each row to its nearest node index."""
        # chunked to keep memory bounded on large images
        out = np.empty(len(data), dtype="int32")
        step = 100_000
        for i in range(0, len(data), step):
            chunk = data[i:i + step]
            d = np.sum(
                (chunk[:, None, :] - self.weights[None, :, :]) ** 2, axis=2
            )
            out[i:i + step] = np.argmin(d, axis=1)
        return out

    def components(self):
        """Connected components of the node graph -> {node: cluster_id}."""
        adj = {i: set() for i in range(len(self.weights))}
        for e in self.edges:
            a, b = tuple(e)
            adj[a].add(b)
            adj[b].add(a)

        label = {}
        cid = 0
        for start in range(len(self.weights)):
            if start in label:
                continue
            stack = [start]
            while stack:
                u = stack.pop()
                if u in label:
                    continue
                label[u] = cid
                stack.extend(adj[u] - label.keys())
            cid += 1
        return label
