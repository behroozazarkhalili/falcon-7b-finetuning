"""Tests for configuration utilities."""

import pytest
import tempfile
from pathlib import Path
from omegaconf import DictConfig
import yaml

from src.utils.config import load_config, merge_configs, save_config, validate_config


class TestConfigUtils:
    """Test configuration utility functions."""
    
    def test_load_config_success(self):
        """Test successful config loading."""
        config_data = {
            "model": {"name": "test-model"},
            "training": {"epochs": 5}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            assert isinstance(config, DictConfig)
            assert config.model.name == "test-model"
            assert config.training.epochs == 5
        finally:
            Path(config_path).unlink()
    
    def test_load_config_file_not_found(self):
        """Test config loading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config("non_existent_file.yaml")
    
    def test_merge_configs(self):
        """Test configuration merging."""
        config1 = DictConfig({"a": 1, "b": {"x": 1}})
        config2 = DictConfig({"b": {"y": 2}, "c": 3})
        
        merged = merge_configs(config1, config2)
        
        assert merged.a == 1
        assert merged.b.x == 1
        assert merged.b.y == 2
        assert merged.c == 3
    
    def test_save_config(self):
        """Test configuration saving."""
        config = DictConfig({"test": "value"})
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_config.yaml"
            save_config(config, output_path)
            
            assert output_path.exists()
            
            # Load and verify
            loaded_config = load_config(output_path)
            assert loaded_config.test == "value"
    
    def test_validate_config_success(self):
        """Test successful config validation."""
        config = DictConfig({"model": {"name": "test"}, "training": {"epochs": 5}})
        required_keys = ["model.name", "training.epochs"]
        
        # Should not raise any exception
        validate_config(config, required_keys)
    
    def test_validate_config_missing_key(self):
        """Test config validation with missing key."""
        config = DictConfig({"model": {"name": "test"}})
        required_keys = ["model.name", "training.epochs"]
        
        with pytest.raises(KeyError):
            validate_config(config, required_keys) 