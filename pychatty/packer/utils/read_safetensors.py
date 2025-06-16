import safetensors

def read_safetensors(file_path) -> dict:
    """
    Reads a safetensors file and returns its content as a dictionary.

    Args:
        file_path (str): The path to the safetensors file.

    Returns:
        dict: A dictionary containing the tensors from the safetensors file.
    """
    return safetensors.torch.load_file(file_path)