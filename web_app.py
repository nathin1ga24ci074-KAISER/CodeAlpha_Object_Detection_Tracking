"""
Standalone Web Application Entry Point — CodeAlpha Task 4: Object Detection and Tracking.
Launches the Flask web server providing the browser interface at http://127.0.0.1:5000.
"""

import argparse
import sys
from src.web.app_server import create_app


def parse_args(args=None):
    """Parses command-line arguments for the web server."""
    parser = argparse.ArgumentParser(
        description="CodeAlpha Object Detection & Tracking — Web Dashboard Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the web server (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to listen on (default: 5000)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable Flask debug mode (default: False)"
    )
    return parser.parse_args(args)


def main():
    """Main execution function launching the Flask development server."""
    args = parse_args()
    print("=" * 64)
    print("  CodeAlpha Object Detection and Tracking — Web Server")
    print(f"  Access URL: http://{args.host}:{args.port}")
    print(f"  Debug Mode: {'Enabled' if args.debug else 'Disabled'}")
    print("=" * 64)

    app = create_app()
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )


if __name__ == "__main__":
    main()
