import numpy as np
import pandas as pd

from src.foundation_model.data.dataset import CompoundDataset


def test_zero_rate_removes_attribute():
    # Create test data
    descriptor = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})
    attributes = pd.DataFrame(
        {"attr1": [0.1, 0.2, 0.3], "attr2": [0.4, 0.5, 0.6], "attr3": [0.7, 0.8, 0.9]}
    )

    # Test with rate=0
    dataset = CompoundDataset(
        descriptor=descriptor,
        attributes=attributes,
        attr1=1.0,
        attr2=0.0,  # This should be removed
        attr3=1.0,
    )

    # Check that attr2 was removed
    assert len(dataset.attributes) == 2
    assert "attr2" not in dataset.attributes
    assert dataset.y.shape[1] == 2
    assert dataset.mask.shape[1] == 2


def test_filter_attributes_true():
    # Create test data
    descriptor = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})
    attributes = pd.DataFrame(
        {"attr1": [0.1, 0.2, 0.3], "attr2": [0.4, 0.5, 0.6], "attr3": [0.7, 0.8, 0.9]}
    )

    # Test with filter_attributes=True
    dataset = CompoundDataset(
        descriptor=descriptor,
        attributes=attributes,
        filter_attributes=True,
        attr1=1.0,  # Only specify two attributes
        attr3=1.0,
    )

    # Check that only specified attributes are kept
    assert len(dataset.attributes) == 2
    assert set(dataset.attributes) == {"attr1", "attr3"}
    assert dataset.y.shape[1] == 2
    assert dataset.mask.shape[1] == 2


def test_filter_attributes_false():
    # Create test data
    descriptor = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})
    attributes = pd.DataFrame(
        {"attr1": [0.1, 0.2, 0.3], "attr2": [0.4, 0.5, 0.6], "attr3": [0.7, 0.8, 0.9]}
    )

    # Test with filter_attributes=False
    dataset = CompoundDataset(
        descriptor=descriptor,
        attributes=attributes,
        filter_attributes=False,
        attr1=1.0,  # Only specify two attributes
        attr3=1.0,
    )

    # Check that all attributes are kept
    assert len(dataset.attributes) == 3
    assert set(dataset.attributes) == {"attr1", "attr2", "attr3"}
    assert dataset.y.shape[1] == 3
    assert dataset.mask.shape[1] == 3


def test_rate_reduces_valid_samples():
    # Create test data
    descriptor = pd.DataFrame({"feat1": [1, 2, 3, 4, 5], "feat2": [6, 7, 8, 9, 10]})
    attributes = pd.DataFrame(
        {
            "attr1": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )

    # Test with rate=0.4 (should keep 2 samples)
    dataset = CompoundDataset(descriptor=descriptor, attributes=attributes, attr1=0.4)

    # Check that the mask has the correct number of 1s
    assert np.sum(dataset.mask) == 2  # 40% of 5 samples = 2 samples


def test_all_combinations():
    # Create test data
    descriptor = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})
    attributes = pd.DataFrame(
        {
            "attr1": [0.1, 0.2, 0.3],
            "attr2": [0.4, 0.5, 0.6],
            "attr3": [0.7, 0.8, 0.9],
            "attr4": [1.0, 1.1, 1.2],
        }
    )

    # Test various combinations
    test_cases = [
        {
            "filter_attributes": True,
            "rates": {"attr1": 1.0, "attr2": 0.0, "attr3": 0.5},
            "expected_attrs": {"attr1", "attr3"},
        },
        {
            "filter_attributes": False,
            "rates": {"attr1": 1.0, "attr2": 0.0, "attr3": 0.5},
            "expected_attrs": {
                "attr1",
                "attr3",
                "attr4",
            },  # attr2 removed due to rate=0
        },
        {
            "filter_attributes": True,
            "rates": {"attr1": 0.0, "attr2": 0.0},
            "expected_attrs": set(),  # All specified attributes have rate=0
        },
    ]

    for case in test_cases:
        dataset = CompoundDataset(
            descriptor=descriptor,
            attributes=attributes,
            filter_attributes=case["filter_attributes"],
            **case["rates"],
        )

        actual_attrs = set(dataset.attributes)
        expected_attrs = case["expected_attrs"]
        print(f"\nTest case: {case}")
        print(f"Actual attributes: {actual_attrs}")
        print(f"Expected attributes: {expected_attrs}")
        assert actual_attrs == expected_attrs, (
            f"Failed for case: {case}\n"
            f"Actual attributes: {actual_attrs}\n"
            f"Expected attributes: {expected_attrs}"
        )


if __name__ == "__main__":
    # Run all tests
    test_zero_rate_removes_attribute()
    test_filter_attributes_true()
    test_filter_attributes_false()
    test_rate_reduces_valid_samples()
    test_all_combinations()
    print("All tests passed!")
