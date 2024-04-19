import yaml


def yaml_to_dict(yaml_file):
    """
    Convert YAML file to dictionary.

    Args:
    - yaml_file (str): Path to the YAML file

    Returns:
    - dict: Dictionary containing YAML data
    """
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return data