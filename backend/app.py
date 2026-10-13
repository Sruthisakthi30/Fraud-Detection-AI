"""
FraudShield AI - Flask Application Entry Point
"""
import logging
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, send_from_directory
from flask_cors import CORS

from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    root = Path(__file__).resolve().parent.parent
    app  = Flask(
        __name__,
        static_folder=str(root / "frontend"),
        static_url_path="",
    )

    # Allow requests from any origin in development
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register API blueprint
    from backend.routes import api
    app.register_blueprint(api)

    # Serve frontend
    @app.route("/")
    def index():
        return send_from_directory(str(root / "frontend"), "index.html")

    @app.route("/<path:path>")
    def static_files(path):
        return send_from_directory(str(root / "frontend"), path)

    logger.info("FraudShield AI app created.")
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False,   # avoids double model loading
    )
