"""Tests for wandb utilities."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from omegaconf import DictConfig

from src.utils.wandb_utils import WandbManager, init_wandb_from_config


class TestWandbManager:
    """Test WandbManager class."""
    
    def setup_method(self):
        """Setup test configuration."""
        self.config = DictConfig({
            "experiment": {
                "name": "test-experiment",
                "tags": ["test", "falcon"],
                "description": "Test experiment"
            },
            "wandb": {
                "project": "test-project",
                "entity": "test-entity",
                "group": "test-group",
                "job_type": "training",
                "settings": {
                    "save_code": True,
                    "watch_model": True,
                    "watch_freq": 100
                },
                "log": {
                    "gradients": False,
                    "parameters": True
                },
                "artifacts": {
                    "log_model_checkpoints": True,
                    "log_final_model": True
                }
            },
            "training": {
                "report_to": ["wandb"]
            }
        })
    
    @patch('src.utils.wandb_utils.wandb')
    def test_init_wandb_success(self, mock_wandb):
        """Test successful wandb initialization."""
        mock_run = Mock()
        mock_run.name = "test-run"
        mock_run.id = "test-id"
        mock_wandb.init.return_value = mock_run
        
        manager = WandbManager(self.config)
        result = manager.init_wandb()
        
        assert result == mock_run
        assert manager.run == mock_run
        mock_wandb.init.assert_called_once()
    
    @patch('src.utils.wandb_utils.wandb')
    def test_init_wandb_failure(self, mock_wandb):
        """Test wandb initialization failure."""
        mock_wandb.init.side_effect = Exception("Wandb init failed")
        
        manager = WandbManager(self.config)
        
        with pytest.raises(Exception):
            manager.init_wandb()
    
    def test_prepare_config_for_wandb(self):
        """Test config preparation for wandb."""
        manager = WandbManager(self.config)
        config_dict = manager._prepare_config_for_wandb()
        
        # Should contain experiment config but not wandb config
        assert "experiment" in config_dict
        assert "wandb" not in config_dict
        assert config_dict["experiment"]["name"] == "test-experiment"
    
    @patch('src.utils.wandb_utils.wandb')
    def test_watch_model(self, mock_wandb):
        """Test model watching functionality."""
        mock_run = Mock()
        mock_wandb.watch = Mock()
        
        manager = WandbManager(self.config)
        manager.run = mock_run
        
        mock_model = Mock()
        manager.watch_model(mock_model)
        
        mock_wandb.watch.assert_called_once_with(
            mock_model, 
            log=["parameters"], 
            log_freq=100
        )
    
    def test_watch_model_no_run(self):
        """Test model watching when no run is active."""
        manager = WandbManager(self.config)
        manager.run = None
        
        mock_model = Mock()
        # Should not raise exception
        manager.watch_model(mock_model)
    
    @patch('src.utils.wandb_utils.wandb')
    def test_log_metrics(self, mock_wandb):
        """Test metrics logging."""
        mock_run = Mock()
        mock_wandb.log = Mock()
        
        manager = WandbManager(self.config)
        manager.run = mock_run
        
        metrics = {"loss": 0.5, "accuracy": 0.8}
        manager.log_metrics(metrics, step=100)
        
        mock_wandb.log.assert_called_once_with(metrics, step=100)
    
    @patch('src.utils.wandb_utils.wandb')
    def test_log_model_info(self, mock_wandb):
        """Test model info logging."""
        mock_run = Mock()
        mock_wandb.config = Mock()
        
        manager = WandbManager(self.config)
        manager.run = mock_run
        
        mock_model = Mock()
        mock_model.parameters.return_value = [Mock(numel=lambda: 1000)]
        
        mock_tokenizer = Mock()
        mock_tokenizer.vocab_size = 50000
        mock_tokenizer.model_max_length = 512
        
        manager.log_model_info(mock_model, mock_tokenizer)
        
        mock_wandb.config.update.assert_called_once()
    
    @patch('src.utils.wandb_utils.wandb')
    def test_log_artifact(self, mock_wandb):
        """Test artifact logging."""
        mock_run = Mock()
        mock_artifact = Mock()
        mock_wandb.Artifact.return_value = mock_artifact
        mock_wandb.log_artifact = Mock()
        
        manager = WandbManager(self.config)
        manager.run = mock_run
        
        manager.log_artifact(
            artifact_path="/path/to/model",
            artifact_name="test-model",
            artifact_type="model",
            description="Test model",
            metadata={"version": "1.0"}
        )
        
        mock_wandb.Artifact.assert_called_once_with(
            name="test-model",
            type="model",
            description="Test model",
            metadata={"version": "1.0"}
        )
        mock_artifact.add_file.assert_called_once_with("/path/to/model")
        mock_wandb.log_artifact.assert_called_once_with(mock_artifact)
    
    @patch('src.utils.wandb_utils.wandb')
    def test_finish(self, mock_wandb):
        """Test wandb run finishing."""
        mock_run = Mock()
        mock_wandb.finish = Mock()
        
        manager = WandbManager(self.config)
        manager.run = mock_run
        
        manager.finish()
        
        mock_wandb.finish.assert_called_once()


class TestInitWandbFromConfig:
    """Test init_wandb_from_config function."""
    
    def test_wandb_not_in_report_to(self):
        """Test when wandb is not in report_to."""
        config = DictConfig({
            "training": {"report_to": ["tensorboard"]}
        })
        
        result = init_wandb_from_config(config)
        assert result is None
    
    def test_wandb_in_report_to_string(self):
        """Test when report_to is a string containing wandb."""
        config = DictConfig({
            "training": {"report_to": "wandb"},
            "experiment": {"name": "test"},
            "wandb": {"project": "test"}
        })
        
        with patch('src.utils.wandb_utils.WandbManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            result = init_wandb_from_config(config)
            
            mock_manager_class.assert_called_once_with(config)
            mock_manager.init_wandb.assert_called_once_with(resume=None)
            assert result == mock_manager
    
    @patch('src.utils.wandb_utils.wandb', None)
    def test_wandb_not_installed(self):
        """Test when wandb is not installed."""
        config = DictConfig({
            "training": {"report_to": ["wandb"]}
        })
        
        with patch.dict('sys.modules', {'wandb': None}):
            result = init_wandb_from_config(config)
            assert result is None 