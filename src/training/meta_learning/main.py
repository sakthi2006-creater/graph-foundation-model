#!/usr/bin/env python3
"""MAML Meta-Learning CLI entrypoint - 1st-order approximation."""

import argparse
from pathlib import Path

from src.config_loader import create_config_parser, load_config
from src.training.meta_learning.maml_trainer import MAMLTrainer


def main():
    parser = create_config_parser()

    parser.add_argument('--inner_steps', type=int, default=5, help='Inner loop adaptation steps')
    parser.add_argument('--domains', nargs='+', default=None, help='Source domains for meta-train')
    parser.add_argument('--target_domain', type=str, default='amazon_photo', help='Target for meta-test')

    args = parser.parse_args()
    
    config = load_config(str(args.config), args)
    
    trainer = MAMLTrainer(str(args.config), args.inner_steps)
    trainer.meta_fit(
        num_epochs=args.num_epochs,
        source_domains=args.domains,
        target_domain=args.target_domain
    )


if __name__ == '__main__':
    main()

