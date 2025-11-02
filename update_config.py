#!/usr/bin/env python3
"""Script to update verl config file with required parameters."""
import sys
import yaml
import os

def update_config(config_path, config_name):
    """Update config file with required parameters."""
    config_file = os.path.join(config_path, f"{config_name}.yaml")
    
    if not os.path.exists(config_file):
        print(f"Config file not found: {config_file}")
        sys.exit(1)
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add required top-level parameters if not present
    if 'ulysses_sequence_parallel_size' not in config:
        config['ulysses_sequence_parallel_size'] = 1
    if 'use_remove_padding' not in config:
        config['use_remove_padding'] = False
    
    # Ensure model section exists
    if 'model' not in config:
        config['model'] = {}
    if 'fsdp_config' not in config['model']:
        config['model']['fsdp_config'] = {}
    
    # Add required fsdp_config parameters if not present
    fsdp_defaults = {
        'wrap_policy': {'min_num_params': 0},
        'cpu_offload': False,
        'offload_params': False,
        'model_dtype': 'fp32'
    }
    
    for key, default_value in fsdp_defaults.items():
        if key not in config['model']['fsdp_config']:
            config['model']['fsdp_config'][key] = default_value
        # For nested dicts like wrap_policy, merge instead of replace
        elif isinstance(default_value, dict) and isinstance(config['model']['fsdp_config'][key], dict):
            for subkey, subvalue in default_value.items():
                if subkey not in config['model']['fsdp_config'][key]:
                    config['model']['fsdp_config'][key][subkey] = subvalue
    
    # Ensure data section exists
    if 'data' not in config:
        config['data'] = {}
    
    # Handle global_batch_size -> train_batch_size conversion
    if 'global_batch_size' in config['data'] and 'train_batch_size' not in config['data']:
        config['data']['train_batch_size'] = config['data']['global_batch_size']
    
    # Add required data parameters if not present
    data_defaults = {
        'train_batch_size': 256,  # Default if neither global_batch_size nor train_batch_size exists
        'chat_template': None,
        'custom_cls': {'path': None, 'name': None},
        'multiturn': {
            'enable': False,
            'messages_key': 'messages',
            'tools_key': 'tools',
            'enable_thinking_key': 'enable_thinking'
        },
        'use_shm': False,
        'truncation': 'error'
    }
    
    for key, default_value in data_defaults.items():
        if key not in config['data']:
            config['data'][key] = default_value
        # For nested dicts like multiturn, merge instead of replace
        elif isinstance(default_value, dict) and isinstance(config['data'][key], dict):
            for subkey, subvalue in default_value.items():
                if subkey not in config['data'][key]:
                    config['data'][key][subkey] = subvalue
    
    # Save updated config
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Updated config file: {config_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: update_config.py <config_path> <config_name>")
        sys.exit(1)
    
    update_config(sys.argv[1], sys.argv[2])

