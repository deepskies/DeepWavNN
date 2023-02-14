from wavNN.data.nmist_generator import NMISTGenerator
import pytest


def test_sample_size():
    sample_sizes = [640, 640 * 2, 640 * 3]
    batch_size = 64
    generator = NMISTGenerator()(
        sample_size=sample_sizes, split=True, batch_size=batch_size
    )
    for dataset, sample_size in zip(generator, sample_sizes):
        assert len(generator[dataset]) == sample_size / batch_size


def test_data_split():
    generator = NMISTGenerator()(sample_size=[640, 640, 640])
    assert {"training", "validation", "test"} == set(generator.keys())


def test_data_incorrect_split_size():
    with pytest.raises(AssertionError):
        NMISTGenerator()(sample_size=[640, 640], split=True)


def test_data_same_size_split():
    sample_size = 640
    batch_size = 64
    generator = NMISTGenerator()(
        sample_size=sample_size, split=True, batch_size=batch_size
    )
    for dataset in generator:
        assert len(generator[dataset]) == sample_size / batch_size


def test_sample_too_big():
    sample_size = 10**8
    with pytest.raises(AssertionError):
        NMISTGenerator()(sample_size=sample_size, split=False)


def test_data_no_split():
    sample_size = 640
    batch_size = 64
    generator = NMISTGenerator()(
        sample_size=sample_size, split=False, batch_size=batch_size
    )
    assert {"training", "validation", "test"} == set(generator.keys())

    assert generator["test"] is None
    assert generator["validation"] is None
    assert len(generator["training"]) == sample_size / batch_size


def test_batching():
    sample_size = 640
    batch_size = 64
    generator = NMISTGenerator()(
        sample_size=sample_size, split=False, batch_size=batch_size
    )
    training = generator["training"]
    train_features, _ = next(iter(training))

    assert len(train_features[0]) == batch_size


def test_label_alignment():
    # Labels don't shift
    pass
