#!/usr/bin/env python3
"""
SWIFT-LLM Interactive Chat

A simple interactive chat interface demonstrating SWIFT-LLM capabilities.

Usage:
    python chat.py
    
Commands:
    /stats  - Show performance statistics
    /clear  - Clear the cache
    /help   - Show help
    /exit   - Exit the chat
"""

import os
import sys

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from swift_llm import SwiftLLM


def print_welcome():
    """Print welcome message."""
    print("\n" + "=" * 60)
    print("  SWIFT-LLM Interactive Chat")
    print("  Semantic-Aware Intelligent Fast Inference")
    print("=" * 60)
    print("\n  Commands:")
    print("    /stats  - Show performance statistics")
    print("    /clear  - Clear the cache")
    print("    /help   - Show this help message")
    print("    /exit   - Exit the chat")
    print("\n  Type your message and press Enter to chat.")
    print("-" * 60 + "\n")


def format_response(response) -> str:
    """Format response with metadata."""
    lines = [
        f"\n[Assistant]: {response.text}",
        "",
        f"   [{response.total_latency_ms:.0f}ms | "
        f"{'Cached' if response.cache_hit else response.model_used} | "
        f"{response.confidence_score:.0%} conf | "
        f"${response.estimated_cost:.4f}]",
    ]
    return "\n".join(lines)


def main():
    # Check for API keys
    if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n[Error] No API keys found.")
        print("   Please set at least one of these environment variables:")
        print("   - GROQ_API_KEY (recommended, free at https://console.groq.com/)")
        print("   - OPENAI_API_KEY")
        print("\n   Example:")
        print("   export GROQ_API_KEY='your-key-here'")
        print()
        sys.exit(1)
    
    print_welcome()
    
    # Initialize SWIFT-LLM
    print("[Initializing SWIFT-LLM...]")
    swift = SwiftLLM()
    print("[Ready]\n")
    
    # Chat history for multi-turn
    chat_history = []
    
    while True:
        try:
            # Get user input
            user_input = input("[You]: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower()
                
                if cmd == "/exit" or cmd == "/quit":
                    print("\n[Goodbye]\n")
                    swift.save_cache()
                    break
                
                elif cmd == "/stats":
                    print("\n" + "-" * 40)
                    swift.print_stats()
                    print("-" * 40 + "\n")
                    continue
                
                elif cmd == "/clear":
                    swift.clear_cache()
                    chat_history.clear()
                    print("[Cache and history cleared]\n")
                    continue
                
                elif cmd == "/help":
                    print("\n  Commands:")
                    print("    /stats  - Show performance statistics")
                    print("    /clear  - Clear cache and history")
                    print("    /help   - Show this help message")
                    print("    /exit   - Exit the chat\n")
                    continue
                
                else:
                    print(f"[Unknown command: {cmd}]")
                    print("   Type /help for available commands.\n")
                    continue
            
            # Add to history
            chat_history.append({"role": "user", "content": user_input})
            
            # Generate response
            if len(chat_history) == 1:
                # Single query mode
                response = swift.query(user_input)
            else:
                # Multi-turn mode
                response = swift.chat(chat_history)
            
            # Add assistant response to history
            chat_history.append({"role": "assistant", "content": response.text})
            
            # Print response
            print(format_response(response))
            print()
            
        except KeyboardInterrupt:
            print("\n\n[Goodbye]\n")
            swift.save_cache()
            break
        
        except Exception as e:
            print(f"\n[Error]: {e}\n")


if __name__ == "__main__":
    main()
