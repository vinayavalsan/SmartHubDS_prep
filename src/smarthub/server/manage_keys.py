"""CLI to manage SmartHub API keys: issue, revoke, and list.

Examples
--------
    # Issue a key for Anton (prints the raw key ONCE — copy it now):
    python -m smarthub.server.manage_keys create --client anton --note "prod bid client"

    # List issued keys (metadata only; never shows the raw key):
    python -m smarthub.server.manage_keys list

    # Revoke by key id or by client name:
    python -m smarthub.server.manage_keys revoke --key-id a1b2c3d4e5f6
    python -m smarthub.server.manage_keys revoke --client anton

Reads the same DB the serving API uses (``SMARTHUB_PREDICTION_LOG_DB_URL``, or
``SMARTHUB_AUTH_DB_URL`` to override). Run it wherever that DB is reachable —
e.g. inside the ``serve`` container.
"""

from __future__ import annotations

import argparse
import sys

from smarthub.server.auth import ApiKeyStore


def _cmd_create(store: ApiKeyStore, args: argparse.Namespace) -> int:
    """Issue a key and print it once."""
    raw_key, key_id = store.create_key(
        args.client, note=args.note, expires_in_days=args.expires_in_days
    )
    print(f"client:  {args.client}")
    print(f"key_id:  {key_id}")
    print(f"api_key: {raw_key}")
    if args.expires_in_days is not None:
        print(f"expires: in {args.expires_in_days} day(s)")
    else:
        print("expires: never")
    print(
        "\nStore this key now — it is shown only once and cannot be recovered.\n"
        "Send it to the client over a secure channel (password manager / not "
        "plain chat)."
    )
    return 0


def _cmd_revoke(store: ApiKeyStore, args: argparse.Namespace) -> int:
    """Deactivate keys by id or client name."""
    n = store.revoke(key_id=args.key_id, client_name=args.client)
    target = args.key_id or args.client
    print(f"Revoked {n} key(s) for {target!r}.")
    return 0 if n else 1


def _cmd_list(store: ApiKeyStore, args: argparse.Namespace) -> int:
    """Print issued keys (metadata only)."""
    keys = store.list_keys(active_only=args.active_only)
    if not keys:
        print("No keys found.")
        return 0
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for k in keys:
        exp = k.get("expires_at")
        if not k["active"]:
            state = "revoked"
        elif exp is not None and now >= (
            exp.replace(tzinfo=None) if exp.tzinfo else exp
        ):
            state = "expired"
        else:
            state = "active"
        exp_txt = f" exp={exp}" if exp else ""
        note = f" — {k['note']}" if k.get("note") else ""
        print(
            f"{k['key_id']}  {k['client_name']:<20} {state:<8} "
            f"{k['created_at']}{exp_txt}{note}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the key-management CLI."""
    parser = argparse.ArgumentParser(
        prog="smarthub.server.manage_keys",
        description="Issue, revoke, and list SmartHub API keys.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Issue a new API key for a client.")
    p_create.add_argument("--client", required=True, help="Client/consumer name.")
    p_create.add_argument("--note", default=None, help="Optional note (purpose/owner).")
    p_create.add_argument(
        "--expires-in-days",
        type=float,
        default=None,
        help="Expire the key this many days from now (omit = never expires).",
    )
    p_create.set_defaults(func=_cmd_create)

    p_revoke = sub.add_parser("revoke", help="Revoke key(s) by id or client name.")
    p_revoke.add_argument("--key-id", default=None, help="Key id to revoke.")
    p_revoke.add_argument(
        "--client", default=None, help="Revoke all of a client's keys."
    )
    p_revoke.set_defaults(func=_cmd_revoke)

    p_list = sub.add_parser("list", help="List issued keys (metadata only).")
    p_list.add_argument(
        "--active-only", action="store_true", help="Show only active keys."
    )
    p_list.set_defaults(func=_cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    if args.command == "revoke" and not (args.key_id or args.client):
        print("revoke needs --key-id or --client", file=sys.stderr)
        return 2
    store = ApiKeyStore()
    return int(args.func(store, args))


if __name__ == "__main__":
    raise SystemExit(main())
