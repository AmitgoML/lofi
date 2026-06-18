import pytest

from lucy.agents.common.models import ChatDeps, SaveFileOutput, UserOrgProfile
from lucy.agents.image_agent import ImageAgent


@pytest.mark.asyncio
async def test_save_image_metadata_best_effort_sets_asset_id_on_insert(monkeypatch):
    # Make to_thread execute inline (deterministic)
    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr("lucy.agents.image_agent.asyncio.to_thread", fake_to_thread)
    monkeypatch.setattr("lucy.agents.image_agent.os.path.getsize", lambda _p: 123)

    monkeypatch.setattr(
        "lucy.agents.image_agent.insert_generated_creative_asset_metadata",
        lambda **_kw: {"asset_id": "asset-1"},
    )

    ctx = type(
        "Ctx",
        (),
        {
            "deps": ChatDeps(
                user_id="u1",
                user_profiles=[UserOrgProfile(org_id="o1")],
            )
        },
    )()
    uploaded = SaveFileOutput(
        file_name="x.png", file_path="users/u1/x.png", file_type="image/png"
    )

    await ImageAgent._save_image_metadata_best_effort(
        ctx, uploaded, tmp_path="/tmp/x.png"
    )
    assert uploaded.asset_id == "asset-1"


@pytest.mark.asyncio
async def test_save_image_metadata_best_effort_sets_asset_id_on_version_insert(
    monkeypatch,
):
    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr("lucy.agents.image_agent.asyncio.to_thread", fake_to_thread)
    monkeypatch.setattr("lucy.agents.image_agent.os.path.getsize", lambda _p: 123)

    monkeypatch.setattr(
        "lucy.agents.image_agent.insert_generated_creative_asset_version",
        lambda **_kw: {"asset_id": "asset-2"},
    )

    ctx = type(
        "Ctx",
        (),
        {
            "deps": ChatDeps(
                user_id="u1",
                user_profiles=[UserOrgProfile(org_id="o1")],
            )
        },
    )()
    uploaded = SaveFileOutput(
        file_name="y.png", file_path="users/u1/y.png", file_type="image/png"
    )

    await ImageAgent._save_image_metadata_best_effort(
        ctx, uploaded, tmp_path="/tmp/y.png", parent_asset_id="parent-1"
    )
    assert uploaded.asset_id == "asset-2"
