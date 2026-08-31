from api.chute.update_policy import plan_external_chute_update


def test_omitted_defaults_do_not_update_external_chute():
    updates, unsupported = plan_external_chute_update(
        {
            "tagline": "",
            "readme": "",
            "tool_description": "",
            "logo_id": None,
            "max_instances": None,
            "scaling_threshold": None,
            "shutdown_after_seconds": 300,
            "disabled": True,
        },
        {"disabled"},
    )

    assert updates == {"disabled": True}
    assert unsupported == ()


def test_explicit_catalog_clears_are_preserved():
    updates, unsupported = plan_external_chute_update(
        {
            "tagline": "",
            "readme": None,
            "tool_description": None,
            "logo_id": "",
        },
        {"tagline", "readme", "tool_description", "logo_id"},
    )

    assert updates == {
        "tagline": "",
        "readme": "",
        "tool_description": None,
        "logo_id": None,
    }
    assert unsupported == ()


def test_explicit_hosted_scaling_fields_are_rejected_even_at_defaults():
    updates, unsupported = plan_external_chute_update(
        {
            "max_instances": None,
            "scaling_threshold": None,
            "shutdown_after_seconds": 300,
        },
        {"max_instances", "scaling_threshold", "shutdown_after_seconds"},
    )

    assert updates == {}
    assert unsupported == (
        "max_instances",
        "scaling_threshold",
        "shutdown_after_seconds",
    )
