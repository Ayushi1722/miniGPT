import argparse
import torch

from minigpt.model.config import get_default_config
from minigpt.model.gpt import GPT
from minigpt.data.tokenizer import get_encoder
from minigpt.data.datasets import get_shakespeare_dataset
from minigpt.training.trainer import train, generate_text

def main():
    parser = argparse.ArgumentParser(description="Train and generate text with MiniGPT")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--generate", action="store_true", help="Generate text from a prompt")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt for text generation")
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.9, help="Temperature for sampling")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--model_path", type=str, default="minigpt_model.pt", help="Path to save/load model")
    args = parser.parse_args()
    
    # Get default config
    config = get_default_config()
    config.epochs = args.epochs
    
    # Initialize tokenizer
    tokenizer = get_encoder()
    
    if args.train:
        print("Initializing model...")
        model = GPT(config).to(config.device)
        
        print("Loading dataset...")
        dataset = get_shakespeare_dataset(tokenizer, config)
        
        print("Starting training...")
        model = train(model, dataset, config)
        
        print(f"Saving model to {args.model_path}")
        torch.save(model.state_dict(), args.model_path)
    
    if args.generate:
        # Load model if not already trained in this session
        if not args.train or not model:
            print(f"Loading model from {args.model_path}")
            model = GPT(config).to(config.device)
            try:
                model.load_state_dict(torch.load(args.model_path))
            except FileNotFoundError:
                print(f"Model file {args.model_path} not found. Please train the model first.")
                return
        
        print("Generating text...")
        generated_text = generate_text(
            model, 
            tokenizer, 
            args.prompt, 
            max_new_tokens=args.max_tokens, 
            temperature=args.temperature
        )
        
        print("\n--- PROMPT ---")
        print(args.prompt)
        print("--- GENERATED ---")
        print(generated_text)

if __name__ == "__main__":
    main()
