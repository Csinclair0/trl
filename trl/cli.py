#!/usr/bin/env python
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys

import pkg_resources

from .scripts.utils import TrlParser
from .scripts.env import main as env_main
from .scripts.env import make_parser as make_env_parser
from .scripts.profile import main as profile_main
from .scripts.profile import make_parser as make_profile_parser
from .scripts.vllm_serve import main as vllm_serve_main
from .scripts.vllm_serve import make_parser as make_vllm_serve_parser
from .scripts.vllm_serve_dp import main as vllm_serve_dp_main
from .scripts.vllm_serve_dp import make_parser as make_vllm_serve_dp_parser


trl_version = pkg_resources.get_distribution("trl").version


def main():
    """Main entry point for the CLI."""
    # Create a parser
    parser = TrlParser(prog="TRL CLI", usage="trl", allow_abbrev=False)

    # Create a subparsers object to add subcommands to the parser
    subparsers = parser.add_subparsers(dest="command", help="available commands", parser_class=TrlParser)

    make_env_parser(subparsers)
    make_profile_parser(subparsers)
    make_vllm_serve_parser(subparsers)
    make_vllm_serve_dp_parser(subparsers)

    # Add version option and info
    parser.add_argument("--version", action="version", version=f"%(prog)s {trl_version}")

    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # Execute the appropriate command
    if args.command == "env":
        env_main([])
    elif args.command == "profile":
        profile_main([])
    elif args.command == "vllm-serve":
        # The vLLM serve command requires extra parsing to handle the model argument
        vllm_args = parser.parse_args_and_config()[0]
        vllm_serve_main(vllm_args)
    elif args.command == "vllm-serve-dp":
        # The vLLM serve data parallel command requires extra parsing to handle the model argument
        vllm_dp_args = parser.parse_args_and_config()[0]
        vllm_serve_dp_main(vllm_dp_args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
