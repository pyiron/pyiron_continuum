import requests
import yaml
import warnings


def get_elasticity(
    sub_folder="elastic",
    repo_owner="damask-multiphysics",
    repo_name="DAMASK",
    directory_path="examples/config/phase/mechanical",
):
    """
    Fetches all the elasticity YAML files in the specified directory from the specified GitHub repository.

    Args:
        sub_folder (str): The subfolder within the directory to fetch the YAML files from.
        repo_owner (str): The owner of the GitHub repository.
        repo_name (str): The name of the GitHub repository.
        directory_path (str): The path to the directory containing the YAML files.

    Returns:
        dict: A dictionary containing the YAML content of each file in the directory
    """
    return get_yaml(sub_folder, repo_owner, repo_name, directory_path)


def get_plasticity(
    sub_folder="plastic",
    repo_owner="damask-multiphysics",
    repo_name="DAMASK",
    directory_path="examples/config/phase/mechanical",
):
    """
    Fetches all the plasticity YAML files in the specified directory from the specified GitHub repository.

    Args:
        sub_folder (str): The subfolder within the directory to fetch the YAML files from.
        repo_owner (str): The owner of the GitHub repository.
        repo_name (str): The name of the GitHub repository.
        directory_path (str): The path to the directory containing the YAML files.

    Returns:
        dict: A dictionary containing the YAML content of each file in the directory
    """
    return get_yaml(sub_folder, repo_owner, repo_name, directory_path)


def get_yaml(
    sub_folder="",
    repo_owner="damask-multiphysics",
    repo_name="DAMASK",
    directory_path="examples/config/phase/mechanical",
):
    """
    Fetches all the YAML files in the specified directory from the specified GitHub repository.

    Args:
        sub_folder (str): The subfolder within the directory to fetch the YAML files from.
        repo_owner (str): The owner of the GitHub repository.
        repo_name (str): The name of the GitHub repository.
        directory_path (str): The path to the directory containing the YAML files.

    Returns:
        dict: A dictionary containing the YAML content of each file in the directory
    """

    # GitHub API URL to get the directory contents
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{directory_path}/{sub_folder}"

    # Dictionary to store YAML content
    yaml_dicts = {}

    # Fetch directory contents
    response = requests.get(api_url)

    if response.status_code == 200:
        files = response.json()
        for file in files:
            if file["name"].endswith(".yaml"):
                # Get raw file URL
                raw_url = file["download_url"]

                # Download the file
                file_response = requests.get(raw_url)
                if file_response.status_code == 200:
                    try:
                        # Load the YAML content into a Python dictionary
                        yaml_content = yaml.safe_load(file_response.text)
                        yaml_dicts[file["name"].replace(".yaml", "")] = yaml_content
                    except yaml.YAMLError as e:
                        warnings.warn(f"Failed to load {file['name']}: {e}")
                else:
                    warnings.warn(
                        f"Failed to download {file['name']}: {file_response.status_code}"
                    )
    else:
        response.raise_for_status()
    return yaml_dicts
