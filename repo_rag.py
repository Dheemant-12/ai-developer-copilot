from github_reader import (
    get_repo_contents,
    get_file_content
)


def load_repository_code(
    repo_url
):

    files = get_repo_contents(
        repo_url
    )

    repository_text = ""

    for file in files:

        if (
            file["download_url"]
            is not None
        ):

            content = get_file_content(
                file["download_url"]
            )

            repository_text += (
                f"\n\nFILE: "
                f"{file['name']}\n"
            )

            repository_text += (
                content
            )

    return repository_text