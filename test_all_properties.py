from typing import Optional, Dict, List, Tuple
import torch
import pandas as pd
from pathlib import Path
from multi_task import scaling_laws_test, CompoundDataset

def test_all_properties(
    descriptor_df: pd.DataFrame,
    property_df: pd.DataFrame,
    device: str = 'cpu'
) -> Dict[str, Dict[str, Tuple[List[float], List[float], List[float]]]]:
    """
    Test scaling laws for all properties in both vary_target and vary_others modes.
    
    Args:
        descriptor_df: DataFrame containing input features
        property_df: DataFrame containing property values
        device: Device to run the model on ('cpu' or 'cuda')
    
    Returns:
        Dict with structure:
        {
            property_name: {
                'vary_target': (test_losses, test_losses_std, fractions),
                'vary_others': (test_losses, test_losses_std, fractions)
            }
        }
        where:
        - test_losses: List of mean test losses for each fraction
        - test_losses_std: List of standard deviations of test losses
        - fractions: List of data fractions used
    """
    results: Dict[str, Dict[str, Tuple[List[float], List[float], List[float]]]] = {}
    
    # Create results directory
    results_dir = Path('images/multi_tasks/scaling_laws')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Test each property
    for property_name in CompoundDataset.ATTRIBUTES:
        results[property_name] = {}
        print(f"\n{'='*80}")
        print(f"Testing scaling laws for {property_name}")
        print(f"{'='*80}")
        
        # Test vary_target mode
        print(f"\nMode: vary_target (varying {property_name} data while fixing others)")
        results[property_name]['vary_target'] = scaling_laws_test(
            descriptor=descriptor_df,
            property_data=property_df,
            target_property=property_name,
            mode='vary_target',
            device=device
        )
        
        # Test vary_others mode
        print(f"\nMode: vary_others (fixing {property_name} while varying others)")
        results[property_name]['vary_others'] = scaling_laws_test(
            descriptor=descriptor_df,
            property_data=property_df,
            target_property=property_name,
            mode='vary_others',
            device=device
        )

    return results

if __name__ == '__main__':
    # Example usage
    
    # Check for CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    descriptor_df = pd.read_csv('common_data/composition_dataset.csv')
    property_df = pd.read_csv('common_data/properties_dataset.csv')
    
    # Run tests
    test_all_properties(descriptor_df, property_df, device=device)
