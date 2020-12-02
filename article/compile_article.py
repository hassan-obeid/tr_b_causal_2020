"""
Command line utility to compile the project's report.
"""
import glob
import os
import subprocess

import click


# Create the function to compile the article
@click.command()
def compile_article():
    """
    Compiles the latex file, `main.tex`.
    """
    # Print a beginning message
    print("Beginning article compilation.")

    # Set the current working directory
    current_directory = os.getcwd()
    article_directory = (
        current_directory
        if current_directory.endswith("article")
        else current_directory + "/article/"
    )
    os.chdir(article_directory)

    # Create the subprocess commands
    main_compile_command = [
        "tectonic",
        "main.tex",
    ]

    # Execute the commands
    subprocess.call(main_compile_command, stderr=subprocess.STDOUT)

    # Print a finished message
    print("Finished compiling article.")


if __name__ == "__main__":
    compile_article()
