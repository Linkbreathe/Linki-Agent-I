"""The final answer must surface the model's real response, not just status."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.widgets import Static

from Linki.cli.tui.app import LinkiTuiApp
from Linki.graph.nodes import final_node


def test_final_node_leads_with_answer() -> None:
    out = final_node(
        {
            "passed": True,
            "attempts": 2,
            "last_actor_summary": "分析：这个网站是艾欧泽亚售楼中心，用于查询房屋信息。",
            "plan_summary": "查阅网站并汇报",
        }
    )
    fa = out["final_answer"]
    assert fa.startswith("分析：这个网站是艾欧泽亚售楼中心")
    # Verification info kept as a compact footer, not the headline.
    assert "Verification passed" in fa
    assert "2 attempt" in fa


def test_final_node_failure_includes_reason() -> None:
    out = final_node(
        {
            "passed": False,
            "attempts": 3,
            "last_actor_summary": "部分结果",
            "last_error": "缺少可信来源",
        }
    )
    fa = out["final_answer"]
    assert "部分结果" in fa
    assert "Not verified: 缺少可信来源" in fa
    assert "Verification failed" in fa


def test_final_node_falls_back_to_plan_when_no_answer() -> None:
    out = final_node({"passed": True, "attempts": 1, "plan_summary": "计划 X"})
    assert out["final_answer"].startswith("计划 X")


def test_tui_final_box_shows_real_answer(tmp_path: Path) -> None:
    async def impl() -> None:
        app = LinkiTuiApp(workspace=tmp_path, initial_task=None)
        async with app.run_test() as pilot:
            answer = "这是模型的真实回答：艾欧泽亚售楼中心是房屋信息查询工具。\n\n— Verification passed · 2 attempt(s)"
            app._handle_event({"type": "final_answer", "route": "workflow", "content": answer})
            await pilot.pause()
            body = app.query_one("#final-body", Static)
            # The final box renders markdown; the source markup must carry the answer.
            assert "艾欧泽亚售楼中心" in body.content.markup

    asyncio.run(impl())
