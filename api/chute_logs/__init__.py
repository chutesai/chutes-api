"""Chute log shipper — validator side.

Ingests pre-registration chute pod logs streamed by the in-guest agent (sek8s
``chute-log-shipper.service``), authenticates each shipment by the VM's registry
mTLS leaf, decides whether the guest should keep capturing or stop, and pushes the
lines to a dedicated in-namespace Loki for a bounded window. See
``docs/specs/chute-log-shipper.md``.
"""
