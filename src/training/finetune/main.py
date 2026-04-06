#!/usr/bin/env python3
"""Few-shot finetuning CLI - adapter on pretrained foundation model."""

import argparse
from pathlib import Path

from src.config_loader import create_config_parser, load_config
from src.training.finetune.finetune_model import FinetuneTrainer


def main():
    parser = create_config_parser()

    parser.add_argument('--pretrain_ckpt', type=Path, default=None, help='Load pretrained checkpoint')
    parser.add_argument('--target_domain', type=str, default='amazon_photo', help='Target domain')

    args = parser.parse_args()
    
    config = load_config(str(args.config), args)
    
    trainer = FinetuneTrainer(
        config_path=str(args.config),
        pretrained_ckpt=args.pretrain_ckpt
    )
    trainer.fit(num_epochs=args.num_epochs, target_domain=args.target_domain)


if __name__ == '__main__':
    main()

