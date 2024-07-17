#!/usr/bin/env python3

import argparse

from input_types.pdf import process_pdf
from input_types.txt import process_text_file
from input_types.web import process_web_request


def main():
    parser = argparse.ArgumentParser(description="TLDR summary generator")
    parser.add_argument(
        "-t",
        dest="type",
        choices=["pdf", "txt", "url", "input"],
        help="Specify type of input (pdf, txt, url, input)",
    )
    parser.add_argument(
        "-n",
        dest="inputfilename",
        metavar="inputfilename",
        type=str,
        help="Name of the input file (PDF or text file)",
    )
    parser.add_argument(
        "-w",
        dest="words_per_sentence",
        type=int,
        default=20,
        help="Number of words per sentence for summarization (default: 20)",
    )
    parser.add_argument(
        "-s",
        dest="sentences_per_page",
        type=int,
        default=5,
        help="Number of sentences per page for summarization (default: 5)",
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=str,
        nargs="?",
        const="TLDR.txt",
        help="Save TLDR summaries to a text file (default: TLDR.txt)",
    )

    args = parser.parse_args()

    parse_args(args)


def parse_args(cmd_arg: str) -> None:
    if cmd_arg.type == "pdf":
        process_pdf(
            cmd_arg.inputfilename,
            cmd_arg.words_per_sentence,
            cmd_arg.sentences_per_page,
            cmd_arg.output_file,
        )
    elif cmd_arg.type == "txt":
        process_text_file(
            cmd_arg.inputfilename,
            cmd_arg.words_per_sentence,
            cmd_arg.sentences_per_page,
            cmd_arg.output_file,
        )
    elif cmd_arg.type == "url":
        process_web_request(
            cmd_arg.inputfilename,
            cmd_arg.words_per_sentence,
            cmd_arg.sentences_per_page,
            cmd_arg.output_file,
        )

    else:
        print("Invalid type specified. Use '-t <type>' to process a PDF or text file.")


if __name__ == "__main__":
    main()
