"""Phase 3 training smoke tests."""

import pytest
import torch
from src.config_loader import load_config
from src.training.pretrain.pretrain_model import PretrainTrainer


@pytest.fixture(scope='module')
def config():
    config = load_config()
    config['training']['batch_size'] = 32  # Small for test
    config['training']['max_epochs'] = 1
    return config


def test_pretrain_trainer_init(config):
    """Test PretrainTrainer initialization."""

    trainer = PretrainTrainer()
    assert trainer.model is not None
    assert len(trainer.train_loader) > 0
    assert trainer.device is not None
    print('PretrainTrainer init OK')


def test_pretrain_loss_finite(config):
    """Test 1 batch forward/loss finite."""

    trainer = PretrainTrainer()
    model = trainer.model
    batch = next(iter(trainer.train_loader))
    
    loss = trainer.compute_loss(model, batch)
    assert torch.isfinite(loss), f'Loss not finite: {loss}'
    print(f'Pretrain loss finite: {loss.item():.3f}')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU test')
def test_pretrain_device_cuda(config):
    """Test GPU training step."""

    trainer = PretrainTrainer()
    if trainer.device.type == 'cuda':
        batch = next(iter(trainer.train_loader))
        loss = trainer.compute_loss(trainer.model, batch)
        assert torch.isfinite(loss)
        print('GPU pretrain step OK')


def test_maml_trainer_init(config):
    \"\"\"Test MAMLTrainer initialization.\"\"\"

    trainer = MAMLTrainer()
    assert trainer.model is not None
    print('MAMLTrainer init OK')


def test_maml_loss_finite(config):
    \"\"\"Test MAML inner loop loss finite.\"\"\"

    trainer = MAMLTrainer(inner_steps=1)
    support = trainer.pipeline.load_source_domain('karate')  # Small graph
    query = trainer.pipeline.load_source_domain('karate')
    loss, _ = trainer.inner_loop(trainer.model, support, query, steps=1)
    assert torch.isfinite(loss)
    print('MAML loss finite OK')


def test_finetune_trainer_init(config):
    \"\"\"Test FinetuneTrainer initialization.\"\"\"

    trainer = FinetuneTrainer()
    assert trainer.adapter is not None
    assert not trainer.foundation.encoder[0].requires_grad  # Frozen
    print('FinetuneTrainer init OK')


def test_finetune_loss_finite(config):
    \"\"\"Test finetune loss finite.\"\"\"

    trainer = FinetuneTrainer()
    batch = next(iter(trainer.train_loader))
    loss = trainer.compute_loss(trainer.model, batch)
    assert torch.isfinite(loss)
    print('Finetune loss finite OK')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


