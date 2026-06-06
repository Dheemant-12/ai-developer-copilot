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

                file_info = {
                    "name": item["name"],
                    "download_url": item["download_url"],
                    "size": item["size"]
                }

                files.append(
                    file_info
                )

        return files

    except Exception as e:

        return [
            {
                "name": f"Error: {str(e)}",
                "download_url": None,
                "size": 0
            }
        ]


def get_file_content(download_url):

    try:

        response = requests.get(
            download_url,
            timeout=10
        )

        response.raise_for_status()

        return response.text

    except Exception:

        return ""