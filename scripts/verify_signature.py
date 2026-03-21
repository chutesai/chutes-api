#!/usr/bin/env python3
"""Verify a bittensor signature given an SS58 public key, message, and signature."""

import argparse
import sys

from bittensor_wallet.keypair import Keypair


def main():
    parser = argparse.ArgumentParser(description="Verify a bittensor signature.")
    parser.add_argument("--pubkey", required=True, help="SS58 public key (hotkey address)")
    parser.add_argument("--message", required=True, help="The original message that was signed")
    parser.add_argument("--signature", required=True, help="The signature to verify (hex string)")
    args = parser.parse_args()

    keypair = Keypair(args.pubkey)
    try:
        valid = keypair.verify(args.message, args.signature)
    except Exception as e:
        print(f"Verification failed with error: {e}")
        sys.exit(1)

    if valid:
        print("Signature is VALID.")
    else:
        print("Signature is INVALID.")
        sys.exit(1)


if __name__ == "__main__":
    main()
