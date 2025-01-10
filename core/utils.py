import os
import glob
import inspect

import torch

''' Used in converting the local variables, which are usually an algorithms hyperparameters to a single
dictionary without any nested dictionaries or objects (which are replaced by their class name)'''
def serialize_locals(locals_dict: dict):
    # Unpack dictionaries within locals_dict
    dict_keys = []
    for k in locals_dict:
        if isinstance(locals_dict[k], dict):
            dict_keys.append(k)
    for k in dict_keys:
        nested_dict = locals_dict.pop(k)
        for k_dict in nested_dict:
            locals_dict[k_dict] = nested_dict[k_dict]
    
    # Convert any value that is a class to its name and list to tensor
    for k in locals_dict:
        if inspect.isclass(locals_dict[k]):
            locals_dict[k] = locals_dict[k].__name__

        if isinstance(locals_dict[k], list):
            locals_dict[k] = torch.tensor(locals_dict[k])
    
    return locals_dict

''' Clears existing tensorboard runs that exist within the same directory as the logging directory'''
def clear_logs(log_dir):
    if os.path.exists(log_dir) and os.path.isdir(log_dir):
        print('Warning: run already exists. Deleting previous logs... \n')
        files = glob.glob(os.path.join(log_dir, 'events.*'))
        for f in files:
            try:
                os.remove(f)
            except Exception as e:
                print(f'Failed to delete {f}. Reason {e}')

''' Used to match the length of an input (can be of primative datatype or a list) 
to a reference list'''
def match_to_list(input, ref_list: list):
    if not isinstance(input, list):
        output = [input] * len(ref_list)
    elif len(input) < len(ref_list):
        output = [input[0]] * len(ref_list)
    else:
        output = input
    
    return output
    