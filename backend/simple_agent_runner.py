#!/usr/bin/env python3
"""
CLI entry point for running the SpecTestPilot agent.

Usage:
    # From a file
    python run_agent.py --spec path/to/openapi.yaml
    
    # From stdin
    cat openapi.yaml | python run_agent.py --stdin
    
    # With verbose output
    python run_agent.py --spec path/to/openapi.yaml --verbose
"""

import sys
import json
import argparse
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from spec_test_pilot.graph import run_agent
from spec_test_pilot.schemas import output_to_json, SpecTestPilotOutput


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SpecTestPilot: Generate test cases from OpenAPI specs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a local OpenAPI spec file
    python run_agent.py --spec api.yaml
    
    # Read spec from stdin
    cat api.yaml | python run_agent.py --stdin
    
    # Save output to file
    python run_agent.py --spec api.yaml --output tests.json
    
    # Verbose mode with reward info
    python run_agent.py --spec api.yaml --verbose
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--spec",
        type=str,
        help="Path to OpenAPI spec file (YAML or JSON)"
    )
    input_group.add_argument(
        "--stdin",
        action="store_true",
        help="Read spec from stdin"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty print JSON output (default: True)"
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Compact JSON output (overrides --pretty)"
    )
    
    # Execution options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with reward and timing info"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Custom run ID for tracking"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Read spec content
    if args.stdin:
        spec_text = sys.stdin.read()
    else:
        spec_path = Path(args.spec)
        if not spec_path.exists():
            print(f"Error: Spec file not found: {args.spec}", file=sys.stderr)
            sys.exit(1)
        spec_text = spec_path.read_text()
    
    # Validate input
    if not spec_text.strip():
        print("Error: Empty spec content", file=sys.stderr)
        sys.exit(1)
    
    # Run agent
    if args.verbose:
        print("Running SpecTestPilot agent...", file=sys.stderr)
    
    try:
        result = run_agent(
            spec_text=spec_text,
            run_id=args.run_id,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"Error running agent: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Extract output
    output = result.get("output", {})
    reward = result.get("reward", 0.0)
    intermediate_rewards = result.get("intermediate_rewards", {})
    run_id = result.get("run_id", "")
    
    # Format output
    if args.compact:
        json_output = json.dumps(output)
    else:
        json_output = json.dumps(output, indent=2)
    
    # Write output
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json_output)
        if args.verbose:
            print(f"Output written to: {args.output}", file=sys.stderr)
    else:
        print(json_output)
    
    # Print verbose info
    if args.verbose:
        print("\n" + "=" * 60, file=sys.stderr)
        print("Run Summary", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"Run ID: {run_id}", file=sys.stderr)
        print(f"Final Reward: {reward:.4f}", file=sys.stderr)
        
        if intermediate_rewards:
            print("\nIntermediate Rewards:", file=sys.stderr)
            for name, value in intermediate_rewards.items():
                print(f"  {name}: {value:.4f}", file=sys.stderr)
        
        # Summary stats
        if output:
            spec_summary = output.get("spec_summary", {})
            test_suite = output.get("test_suite", [])
            endpoints = spec_summary.get("endpoints_detected", [])
            
            print(f"\nSpec: {spec_summary.get('title', 'unknown')}", file=sys.stderr)
            print(f"Endpoints detected: {len(endpoints)}", file=sys.stderr)
            print(f"Test cases generated: {len(test_suite)}", file=sys.stderr)
            
            missing_info = output.get("missing_info", [])
            if missing_info:
                print(f"\nMissing info ({len(missing_info)} items):", file=sys.stderr)
                for item in missing_info[:5]:
                    print(f"  - {item}", file=sys.stderr)


def run_on_spec(spec_text: str, verbose: bool = False) -> dict:
    """
    Programmatic interface to run the agent.
    
    Args:
        spec_text: OpenAPI spec content
        verbose: Whether to print verbose output
        
    Returns:
        Dict with output, reward, and metadata
    """
    return run_agent(spec_text=spec_text, verbose=verbose)


def validate_output(output: dict) -> bool:
    """
    Validate output against schema.
    
    Args:
        output: Output dictionary
        
    Returns:
        True if valid, False otherwise
    """
    try:
        SpecTestPilotOutput.model_validate(output)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
