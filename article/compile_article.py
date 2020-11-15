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
        "pdflatex",
        "--interaction=nonstop",
        "main.tex",
    ]
    bibtex_dissertation = [
        "bibtex",
        "main.aux",
    ]
    build_cmds = [
        main_compile_command,
        bibtex_dissertation,
        main_compile_command,
        main_compile_command,
    ]

    # Execute the commands
    for cmd in build_cmds:
        subprocess.call(cmd, stderr=subprocess.STDOUT)

    # Declare endings and base filepaths for unwanted files
    unwanted_endings = [
        ".aux",
        ".lof",
        ".log",
        ".lot",
        ".out",
        ".toc",
        ".bbl",
        ".blg",
        ".tex.bak",
        ".bib.bak",
    ]

    base_filepaths = [
        "./main",
        "../main",
        "sections/",
    ]

    # Cleanup unwanted files
    for ending in unwanted_endings:
        for path in base_filepaths:
            unwanted_file_pattern = path + "*{}".format(ending)
            unwanted_file_list = glob.glob(unwanted_file_pattern)
            for unwanted_file in unwanted_file_list:
                if os.path.exists(unwanted_file):
                    os.remove(unwanted_file)

    # Print a finished message
    print("Finished compiling article.")


if __name__ == "__main__":
    compile_article()
