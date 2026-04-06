#!/usr/bin/env python3
"""Pretraining CLI entrypoint for masked edge prediction + contrastive loss."""

import argparse
from pathlib import Path

from src.config_loader import create_config_parser, load_config
from src.training.pretrain.pretrain_model import PretrainTrainer


def main():
    parser = create_config_parser()

    parser.add_argument('--domains', nargs='+', default=None, help='Source domains (default: all)')

    args = parser.parse_args()
    
    config = load_config(str(args.config), args)
    
    trainer = PretrainTrainer(str(args.config))
    trainer.fit(num_epochs=args.num_epochs)


if __name__ == '__main__':
    main()


