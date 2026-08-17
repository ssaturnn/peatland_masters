import numpy as np

from peatland import gng


def two_blobs(n=400, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.normal([0, 0], 0.05, (n, 2))
    b = rng.normal([3, 3], 0.05, (n, 2))
    return np.vstack([a, b])


def test_gng_fit_grows_and_is_deterministic():
    data = two_blobs()
    net1 = gng.GrowingNeuralGas(max_nodes=20, rng=np.random.default_rng(0))
    net1.fit(data, n_steps=3000)
    net2 = gng.GrowingNeuralGas(max_nodes=20, rng=np.random.default_rng(0))
    net2.fit(data, n_steps=3000)
    assert len(net1.weights) > 2
    # same seed → identical prototypes
    assert np.allclose(net1.weights, net2.weights)


def test_prune_long_edges_splits_two_clusters():
    data = two_blobs()
    net = gng.GrowingNeuralGas(max_nodes=24, rng=np.random.default_rng(0))
    net.fit(data, n_steps=4000)
    net.prune_long_edges(factor=1.0)
    comps = set(net.components().values())
    assert len(comps) >= 2  # the two well-separated blobs separate


def test_predict_covers_all_rows():
    data = two_blobs(n=100)
    net = gng.GrowingNeuralGas(max_nodes=15, rng=np.random.default_rng(0))
    net.fit(data, n_steps=2000)
    labels = net.predict(data)
    assert labels.shape[0] == data.shape[0]
    assert labels.min() >= 0 and labels.max() < len(net.weights)
