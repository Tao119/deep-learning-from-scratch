import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np


def majority_vote(predictions):
    """
    Hard voting: each model contributes one vote (its argmax class).

    Parameters
    ----------
    predictions : list of np.ndarray, each shape (N, K)
        Logit or probability arrays from each model.

    Returns
    -------
    np.ndarray shape (N,)  — winning class per sample
    """
    votes = np.stack([np.argmax(p, axis=1) for p in predictions], axis=1)  # (N, M)
    result = np.apply_along_axis(
        lambda row: np.bincount(row, minlength=predictions[0].shape[1]).argmax(),
        axis=1, arr=votes
    )
    return result


def average_proba(predictions):
    """
    Soft voting: average the probability distributions then argmax.

    Parameters
    ----------
    predictions : list of np.ndarray, each shape (N, K)

    Returns
    -------
    np.ndarray shape (N,)
    """
    stacked = np.stack(predictions, axis=0)   # (M, N, K)
    avg = stacked.mean(axis=0)                # (N, K)
    return np.argmax(avg, axis=1)


def weighted_average(predictions, weights):
    """
    Weighted soft voting.

    Parameters
    ----------
    predictions : list of np.ndarray, each shape (N, K)
    weights     : list or np.ndarray of length M (need not sum to 1)

    Returns
    -------
    np.ndarray shape (N,)
    """
    w = np.array(weights, dtype=float)
    w /= w.sum()
    stacked = np.stack(predictions, axis=0)   # (M, N, K)
    wavg = (stacked * w[:, None, None]).sum(axis=0)  # (N, K)
    return np.argmax(wavg, axis=1)


def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class ModelEnsemble:
    """Holds multiple models and runs ensemble predictions."""

    def __init__(self):
        self._models = []
        self._weights = []
        self._names = []

    def add_model(self, model, weight=1.0, name=None):
        self._models.append(model)
        self._weights.append(weight)
        self._names.append(name or f"model_{len(self._models)}")

    def _collect_probs(self, x, batch_size=64):
        """Return list of (N, K) softmax probability arrays, one per model."""
        all_probs = []
        for model in self._models:
            probs_batches = []
            n = (x.shape[0] // batch_size) * batch_size
            for i in range(0, n, batch_size):
                logits = model.predict(x[i:i+batch_size], train_flg=False)
                probs_batches.append(_softmax(logits))
            all_probs.append(np.vstack(probs_batches))
        return all_probs, n

    def predict(self, x, strategy="weighted", batch_size=64):
        """
        Run ensemble prediction.

        Parameters
        ----------
        x        : np.ndarray (N, C, H, W)
        strategy : 'majority' | 'average' | 'weighted'
        batch_size : int

        Returns
        -------
        np.ndarray (N',)  where N' = (N//batch_size)*batch_size
        """
        probs_list, _ = self._collect_probs(x, batch_size)
        if strategy == "majority":
            return majority_vote(probs_list)
        elif strategy == "average":
            return average_proba(probs_list)
        else:
            return weighted_average(probs_list, self._weights)

    def evaluate(self, x_test, t_test, batch_size=64):
        """
        Compute per-model accuracy and ensemble accuracies.

        Returns
        -------
        dict with keys:
          'per_model'   : {name: acc}
          'majority'    : float
          'average'     : float
          'weighted'    : float
        """
        probs_list, n = self._collect_probs(x_test, batch_size)
        t = t_test[:n]

        per_model = {}
        for name, probs in zip(self._names, probs_list):
            acc = np.mean(np.argmax(probs, axis=1) == t)
            per_model[name] = float(acc)

        def acc(preds):
            return float(np.mean(preds == t))

        return {
            "per_model": per_model,
            "majority":  acc(majority_vote(probs_list)),
            "average":   acc(average_proba(probs_list)),
            "weighted":  acc(weighted_average(probs_list, self._weights)),
        }


def _load_vgglike(output_size=10):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from models.vgg_like import VGGLike
    model = VGGLike(input_channels=3, input_size=32, output_size=output_size)
    pkl = os.path.join(os.path.dirname(__file__),
                       "../experiments/cifar10/C01-vgglike-baseline/VGGLike_cifar10_mild_best.pkl")
    if os.path.exists(pkl):
        import pickle
        with open(pkl, "rb") as f:
            params = pickle.load(f)
        model.params = params
        from collections import OrderedDict
        from common.conv_layers import Convolution, Pooling
        from common.layers import Affine, Relu, SoftmaxWithLoss
        from common.layers_ext import Dropout
        p = model.params
        model.layers = OrderedDict([
            ("Conv1",   Convolution(p["W1"], p["b1"], stride=1, pad=1)),
            ("Relu1",   Relu()),
            ("Conv2",   Convolution(p["W2"], p["b2"], stride=1, pad=1)),
            ("Relu2",   Relu()),
            ("Pool1",   Pooling(2, 2, stride=2)),
            ("Conv3",   Convolution(p["W3"], p["b3"], stride=1, pad=1)),
            ("Relu3",   Relu()),
            ("Conv4",   Convolution(p["W4"], p["b4"], stride=1, pad=1)),
            ("Relu4",   Relu()),
            ("Pool2",   Pooling(2, 2, stride=2)),
            ("Conv5",   Convolution(p["W5"], p["b5"], stride=1, pad=1)),
            ("Relu5",   Relu()),
            ("Conv6",   Convolution(p["W6"], p["b6"], stride=1, pad=1)),
            ("Relu6",   Relu()),
            ("Pool3",   Pooling(2, 2, stride=2)),
            ("Affine1", Affine(p["W7"], p["b7"])),
            ("Relu7",   Relu()),
            ("Drop1",   Dropout(0.5)),
            ("Affine2", Affine(p["W8"], p["b8"])),
        ])
        print(f"Loaded VGGLike from {pkl}")
        return model, "VGGLike(trained)"
    print("VGGLike pkl not found — using random weights.")
    return model, "VGGLike(random)"


def _load_vggbn(output_size=10):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from models.vgg_bn import VGGWithBN
    model = VGGWithBN(input_channels=3, input_size=32, output_size=output_size)
    pkl = os.path.join(os.path.dirname(__file__),
                       "../experiments/cifar10/C02-vggbn-batchnorm/VGGWithBN_cifar10_mild_best.pkl")
    if os.path.exists(pkl):
        import pickle
        with open(pkl, "rb") as f:
            params = pickle.load(f)
        model.params = params
        from collections import OrderedDict
        from common.conv_layers import Convolution, Pooling
        from common.layers import Affine, Relu, SoftmaxWithLoss
        from common.layers_ext import BatchNormalization, Dropout
        p = model.params

        def make_bn(i):
            return BatchNormalization(p[f"gamma{i}"], p[f"beta{i}"])

        model.layers = OrderedDict([
            ("Conv1",   Convolution(p["W1"], p["b1"], stride=1, pad=1)),
            ("BN1",     make_bn(1)),
            ("Relu1",   Relu()),
            ("Conv2",   Convolution(p["W2"], p["b2"], stride=1, pad=1)),
            ("BN2",     make_bn(2)),
            ("Relu2",   Relu()),
            ("Pool1",   Pooling(2, 2, stride=2)),
            ("Conv3",   Convolution(p["W3"], p["b3"], stride=1, pad=1)),
            ("BN3",     make_bn(3)),
            ("Relu3",   Relu()),
            ("Conv4",   Convolution(p["W4"], p["b4"], stride=1, pad=1)),
            ("BN4",     make_bn(4)),
            ("Relu4",   Relu()),
            ("Pool2",   Pooling(2, 2, stride=2)),
            ("Conv5",   Convolution(p["W5"], p["b5"], stride=1, pad=1)),
            ("BN5",     make_bn(5)),
            ("Relu5",   Relu()),
            ("Conv6",   Convolution(p["W6"], p["b6"], stride=1, pad=1)),
            ("BN6",     make_bn(6)),
            ("Relu6",   Relu()),
            ("Pool3",   Pooling(2, 2, stride=2)),
            ("Affine1", Affine(p["W7"], p["b7"])),
            ("Relu7",   Relu()),
            ("Drop1",   Dropout(0.5)),
            ("Affine2", Affine(p["W8"], p["b8"])),
        ])
        print(f"Loaded VGGWithBN from {pkl}")
        return model, "VGGWithBN(trained)"
    print("VGGWithBN pkl not found — using random weights.")
    return model, "VGGWithBN(random)"


def _load_mobilenet(output_size=10):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from models.mobilenet import MobileNet
    model = MobileNet(input_channels=3, output_size=output_size)
    pkl = os.path.join(os.path.dirname(__file__),
                       "../experiments/cifar10/C05-mobilenet-mild/MobileNet_cifar10_mild_best.pkl")
    if os.path.exists(pkl):
        import pickle
        with open(pkl, "rb") as f:
            params = pickle.load(f)
        # MobileNet.update() references params via get_params_and_grads;
        # inject weights by iterating named tensors in order.
        p_ref, _ = model.get_params_and_grads()
        for key in p_ref:
            if key in params:
                p_ref[key][:] = params[key]
        print(f"Loaded MobileNet from {pkl}")
        return model, "MobileNet(trained)"
    print("MobileNet pkl not found — using random weights.")
    return model, "MobileNet(random)"


def _demo():
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from dataset.cifar10 import load_cifar10, normalize

    (x_train, _), (x_test, t_test) = load_cifar10()
    x_train_n, x_test_n, _, _ = normalize(x_train, x_test)

    n_samples = 1000
    x_eval = x_test_n[:n_samples]
    t_eval = t_test[:n_samples]

    vgglike, name1 = _load_vgglike(output_size=10)
    vggbn,   name2 = _load_vggbn(output_size=10)
    mobile,  name3 = _load_mobilenet(output_size=10)

    ensemble = ModelEnsemble()
    ensemble.add_model(vgglike, weight=1.0, name=name1)
    ensemble.add_model(vggbn,   weight=2.0, name=name2)
    ensemble.add_model(mobile,  weight=1.5, name=name3)

    print(f"\nEvaluating on {n_samples} CIFAR-10 test samples ...")
    results = ensemble.evaluate(x_eval, t_eval, batch_size=64)

    header = f"{'Model':<25} {'Accuracy':>10}"
    print("\n" + header)
    print("-" * len(header))
    for name, acc in results["per_model"].items():
        print(f"{name:<25} {acc:>10.4f}")
    print("-" * len(header))
    print(f"{'Ensemble (majority)':<25} {results['majority']:>10.4f}")
    print(f"{'Ensemble (average)':<25} {results['average']:>10.4f}")
    print(f"{'Ensemble (weighted)':<25} {results['weighted']:>10.4f}")


if __name__ == "__main__":
    _demo()
