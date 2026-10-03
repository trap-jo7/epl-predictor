import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_data as fd  # noqa: E402

BS = {"teams": [{"id": 1, "name": "Man Utd"}, {"id": 2, "name": "Arsenal"}],
      "elements": [{"id": 9, "team": 1, "web_name": "Example", "first_name": "A", "second_name": "B",
                    "element_type": 4, "status": "i", "news": "Knock", "minutes": 900,
                    "goals_scored": 3, "assists": 1, "form": "5.5", "photo": "12345.jpg"}]}


def test_players_use_site_names_and_photo_url():
    p = fd.build_players(BS)["Man United"][0]
    assert p["pos"] == "FWD" and p["st"] == "i" and p["photo"].endswith("p12345.png")


def test_fixtures_sorted_and_finished_dropped():
    names = fd.team_names(BS)
    fx = [{"kickoff_time": "2026-10-11T14:00:00Z", "team_h": 2, "team_a": 1},
          {"kickoff_time": "2026-10-10T11:30:00Z", "team_h": 1, "team_a": 2},
          {"kickoff_time": "2026-08-01T11:30:00Z", "team_h": 1, "team_a": 2, "finished": True},
          {"kickoff_time": None, "team_h": 1, "team_a": 2}]
    out = fd.build_fixtures(fx, names)
    assert [f["home"] for f in out] == ["Man United", "Arsenal"]
