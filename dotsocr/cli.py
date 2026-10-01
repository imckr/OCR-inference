"""
DotsOCR CLI — manage the platform from the command line.

Usage:
    ocr run                  Start the server
    ocr run --port 8080      Start on a custom port
    ocr init-db              Initialize the database
    ocr create-admin         Create an admin user
"""

import argparse
import getpass
import logging
import sys

logger = logging.getLogger("dotsocr.cli")


def cmd_run(args):
    """Start the DotsOCR platform server."""
    import uvicorn

    from dotsocr.config import HOST, PORT

    port = args.port or PORT
    host = args.host or HOST

    print(f"""
    ╔══════════════════════════════════════════════╗
    ║             OCR Inference v1.0               ║
    ║                                              ║
    ║   Dashboard:  http://{host}:{port}           ║
    ║   API Docs:   http://{host}:{port}/api/docs  ║
    ╚══════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "dotsocr.app:app",
        host=host,
        port=port,
        timeout_keep_alive=900,
        reload=args.reload,
    )


def cmd_init_db(args):
    """Initialize the database tables."""
    from dotsocr.database import init_db

    init_db()
    print("✓ Database initialized successfully.")


def cmd_create_admin(args):
    """Create an admin user interactively."""
    from dotsocr.auth.utils import hash_password
    from dotsocr.database import User, get_engine, get_session_factory, init_db

    init_db()
    SessionLocal = get_session_factory()
    db = SessionLocal()

    email = args.email or input("Admin email: ").strip()
    name = args.name or input("Admin name: ").strip()
    password = args.password or getpass.getpass("Admin password: ")

    existing = db.query(User).filter(User.email == email).first()
    if existing:
        if existing.is_admin:
            print(f"✗ Admin user {email} already exists.")
            sys.exit(1)
        existing.is_admin = True
        db.commit()
        print(f"✓ User {email} promoted to admin.")
    else:
        user = User(
            email=email,
            name=name,
            hashed_password=hash_password(password),
            is_admin=True,
        )
        db.add(user)
        db.commit()
        print(f"✓ Admin user {email} created.")

    db.close()


def main():
    parser = argparse.ArgumentParser(
        prog="ocr",
        description="DotsOCR — Self-hosted OCR inference platform",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run
    run_parser = subparsers.add_parser("run", help="Start the platform server")
    run_parser.add_argument("--host", type=str, default=None, help="Bind host")
    run_parser.add_argument("--port", type=int, default=None, help="Bind port")
    run_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # init-db
    subparsers.add_parser("init-db", help="Initialize the database")

    # create-admin
    admin_parser = subparsers.add_parser("create-admin", help="Create an admin user")
    admin_parser.add_argument("--email", type=str, help="Admin email")
    admin_parser.add_argument("--name", type=str, help="Admin name")
    admin_parser.add_argument("--password", type=str, help="Admin password")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "init-db":
        cmd_init_db(args)
    elif args.command == "create-admin":
        cmd_create_admin(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
