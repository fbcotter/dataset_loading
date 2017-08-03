#!/usr/bin/env python

import argparse
import pytest


def parse_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args


def main(args=None):
    pytest.main()


if __name__ == '__main__':
    args = parse_args()
    main(args)
