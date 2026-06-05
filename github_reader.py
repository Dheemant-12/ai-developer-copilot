import requests


def get_repo_contents(repo_url):

    try:

        parts = repo_url.rstrip("/").split("/")

        owner = parts[-2]
        repo = parts[-1]

        api_url = (
            f"https://api.github.com/repos/"
            f"{owner}/{repo}/contents"
        )

        response = requests.get(
            api_url,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        files = []

        for item in data:

            if item["type"] == "file":

                files.append(
                    item["name"]
                )

        return files

    except Exception as e:

        return [f"Error: {str(e)}"]