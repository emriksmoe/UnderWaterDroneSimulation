"""Command-line tool for managing trained models"""

import json
from pathlib import Path
from src.utils.config_logger import list_all_models, load_model_config
from src.utils.compare_config import compare_configs_detailed
import argparse


def list_models():
    """List all available models"""
    models = list_all_models()
    
    if not models:
        print("No trained models found!")
        return
    
    print("\n" + "="*100)
    print("📦 AVAILABLE MODELS")
    print("="*100)
    
    for i, model in enumerate(models, 1):
        print(f"\n{i}. {model['name']}")
        print(f"   Path: {model['path']}")
        if 'completed' in model:
            print(f"   Completed: {model['completed']}")
        print(f"   Config: {'✅' if model['has_config'] else '❌'}")
        print(f"   Results: {'✅' if model['has_results'] else '❌'}")


def show_config(model_name: str):
    """Show configuration for a specific model"""
    try:
        config = load_model_config(model_name)
        
        print("\n" + "="*80)
        print(f"⚙️  CONFIGURATION: {model_name}")
        print("="*80)
        
        print("\n💰 REWARD PARAMETERS:")
        for key, value in config['configuration']['reward_parameters'].items():
            if 'reward' in key:
                print(f"   {key}: {value:,.1f}")
        
        print("\n⚠️  PENALTY PARAMETERS:")
        for key, value in config['configuration']['reward_parameters'].items():
            if 'penalty' in key:
                formatted = f"{value:,.4f}" if abs(value) < 1 else f"{value:,.1f}"
                print(f"   {key}: {formatted}")
        
        print("\n🤖 DQN HYPERPARAMETERS:")
        for key, value in config['configuration']['dqn_hyperparameters'].items():
            if isinstance(value, float):
                formatted = f"{value:.6f}" if value < 0.01 else f"{value:.4f}"
            else:
                formatted = f"{value:,}"
            print(f"   {key}: {formatted}")
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")


def main():
    parser = argparse.ArgumentParser(description="Manage trained RL models")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List command
    subparsers.add_parser('list', help='List all available models')
    
    # Show config command
    show_parser = subparsers.add_parser('config', help='Show model configuration')
    show_parser.add_argument('model_name', help='Name of the model')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('models', nargs='+', help='Model names to compare')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_models()
    elif args.command == 'config':
        show_config(args.model_name)
    elif args.command == 'compare':
        compare_configs_detailed(*args.models)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()