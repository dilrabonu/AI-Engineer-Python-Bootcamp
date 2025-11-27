from typing import Dict, Any
import yaml

def load_config(path: str) -> Dict[str, Any]:
    """ Load a YAML configuration file into a dictionary.

    Args: 
        path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration values.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the YAML content is invalid.
    """

    with open(path, "r", encoding="utf-8") as file:
        config: Dict[str, Any] = yaml.safe_load(file)
    return config