import argparse
import logging
from harness.cli import CodeAICLI

def main():
    parser = argparse.ArgumentParser(description="CodeAI Autonomous AI Coding Harness")
    parser.add_argument(
        "--config", 
        type=str, 
        default="codeai.json",
        help="Path to the configuration file (default: codeai.json)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output and logging"
    )
    
    args = parser.parse_args()
    
    logging_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    cli = CodeAICLI(config_path=args.config, verbose=args.verbose)
    cli.run()

if __name__ == "__main__":
    main()
