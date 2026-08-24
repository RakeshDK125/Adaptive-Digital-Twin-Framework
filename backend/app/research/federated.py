from typing import List, Dict
import numpy as np

class FederatedLearningAggregator:
    """
    Simulates Federated Averaging (FedAvg) across multiple Edge Digital Twins.
    """
    
    @staticmethod
    def fed_avg(client_weights: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Aggregates weights from multiple edge twins by averaging them.
        """
        if not client_weights:
            return {}
            
        global_weights = {}
        num_clients = len(client_weights)
        
        # Initialize global weights with zeros
        first_client = client_weights[0]
        for key, tensor in first_client.items():
            global_weights[key] = np.zeros_like(tensor, dtype=np.float32)
            
        # Sum all weights
        for client in client_weights:
            for key, tensor in client.items():
                global_weights[key] += tensor
                
        # Average
        for key in global_weights:
            global_weights[key] /= num_clients
            
        return global_weights
