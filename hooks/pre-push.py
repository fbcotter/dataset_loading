#!/usr/bin/env python

import argparse
import pytest


def parse_args():
    parser = argparse.ArgumenParser()
    args = parser.parse_args()
    return args


def main(args=None):
    pytest.main()


if __name__ == '__main__':
    args = parse_args()
    main(args)
