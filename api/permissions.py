"""
Permissions bitmask stuff.
"""

from pydantic import BaseModel


class Role(BaseModel):
    bitmask: int
    description: str


class Permissioning:
    developer = Role(
        bitmask=1 << 0,
        description="Developer -- create images and chutes on the platform.",
    )
    create_user = Role(
        bitmask=1 << 1,
        description="Create users -- administrator role allowing creation of users.",
    )
    update_user = Role(
        bitmask=1 << 2,
        description="Update users -- administrator role allowing updates to users.",
    )
    delete_user = Role(
        bitmask=1 << 3,
        description="Delete users -- administrator role allowing deletion of users.",
    )
    free_account = Role(
        bitmask=1 << 4,
        description="Free invocation -- run anything, for free.",
    )
    unlimited = Role(
        bitmask=1 << 5,
        description="No rate limits.",
    )
    billing_admin = Role(
        bitmask=1 << 6,
        description="Billing admin.",
    )
    unlimited_dev = Role(
        bitmask=1 << 7,
        description="Unlimited dev activity.",
    )
    invoice_billing = Role(
        bitmask=1 << 8,
        description="Invoice billing option",
    )
    stake_weight_quota = Role(
        bitmask=1 << 9,
        description="Enable stake weighted quota (signed message with the coldkey)",
    )
    subnet_admin = Role(
        bitmask=1 << 10,
        description="Subnet admin",
    )
    public_model_deployment = Role(
        bitmask=1 << 11,
        description="Allow deploying models publicly",
    )
    subnet_invoke = Role(
        bitmask=1 << 12,
        description="Allow invocation of private chutes of a subnet",
    )
    subnet_admin_assign = Role(
        bitmask=1 << 13,
        description="Allow granting subnet admin/request roles to other users, i.e. subnet role self-service",
    )

    @classmethod
    def enabled(cls, user, role):
        return user.permissions_bitmask & role.bitmask == role.bitmask

    @classmethod
    def enable(cls, user, role):
        user.permissions_bitmask |= role.bitmask

    @classmethod
    def disable(cls, user, role):
        user.permissions_bitmask &= ~role.bitmask
