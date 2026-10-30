"""Triage tests built from real inbox examples (offline heuristic + stubbed LLM)."""
import json

from tests.test_pipeline import client, H  # shares the test DB / env setup

AMAZON = {"from": "Amazon.in <order-update@amazon.in>",
          "subject": 'Delivered: "Titan Karishma Quartz..."',
          "body": "Your Orders | Your Account | Buy Again\nYour package was delivered!\nDelivered today\n"
                  "Package was handed to resident.\nOrder #405-7698049-4539522\nTitan Karishma Quartz Analog Watch "
                  "for Men\nTrack package\nWe hope to see you again soon. Amazon.in"}
TLAUNCHER = {"from": "TLauncher Emails Noreply <hello@tlauncher.me>",
             "subject": "[26-10-2026 08:59:01] Password recovery on the site TLauncher",
             "body": "Hey Zacrock,\nA request was received to reset your TLauncher account password!\n"
                     "Select the method of password generation (manual or automatic) and follow the selected link.\n"
                     "Set a new password manually (When clicking on the link, set a new password yourself) "
                     "Or copy and paste this link into your browser: https://tlauncher.org/en/restore?code=abc"}
CHESS = {"from": "Chess.com <hello@chess.com>", "subject": "Your streak is alive!",
         "body": "Keep it going with a quick game or Puzzle before midnight! [image: Streak fire illustration] "
                 "Play a game, do a puzzle, or review a game to add to your streak.\nUnsubscribe | Notification settings"}
SBI = {"from": "cbssbi.cas@alerts.sbi.bank.in", "subject": "E-account statement for your SBI account(s).",
       "body": "Dear Mr. PREETAM MANOJ WAGH,\nYour account(s) details are available at your fingertips. You can view "
               "them online whenever required. For your convenience, please find attached your latest monthly "
               "statement.\nYour e-account statement is protected by a password, which is the last five digits of "
               "customer registered mobile number and date of birth (DOB) in DDMMYY format.\nIn case of any issues, "
               "please lodge a complaint online at contactcentre@sbi.co.in or by calling our 1800 1234 toll free "
               "numbers.\nWith Best Regards,\nTeam SBI\n** Please note: You will need Adobe Acrobat to open the "
               "attachment. Please do not reply to this email as it is a computer generated email."}
COLLEAGUE = {"from": "Rohan Mehta <rohan.mehta@college.edu>", "subject": "DA2 demo slot",
             "body": "Dear Professor,\nCould you confirm whether we can present the DA2 demo on Thursday at 3 pm? "
                     "We need to book the lab by Wednesday.\nThanks,\nRohan"}


def analyze(email, **extra):
    req = {"user_id": "tri", "subject": email["subject"],
           "messages": [{"from": email["from"], "body": email["body"]}], **extra}
    r = client.post("/analyze", headers=H, json=req)
    assert r.status_code == 200, r.text
    return r.json()


def test_amazon_delivery_is_trivial_and_needs_no_reply():
    d = analyze(AMAZON)
    t = d["triage"]
    assert t["topic"] == "shopping_orders" and t["category"] == "trivial" and t["can_delete"]
    assert t["gmail_labels"] == ["AI/Shopping and Orders", "AI/Can delete"]
    assert t["needs_reply"] == "no" and d["draft"] is None and d["draft_skipped"] == "no_reply_needed"
    assert any(k["label"] == "Order number" and "405-7698049-4539522" in k["value"] for k in t["key_details"])
    assert "delivered" in t["sender_wants"].lower()


def test_password_reset_is_security_no_reply():
    t = analyze(TLAUNCHER)["triage"]
    assert t["topic"] == "accounts_security" and t["needs_reply"] == "no"
    assert t["category"] == "important" and t["gmail_labels"] == ["AI/Accounts and Security"]
    assert "only if you" in t["sender_wants"].lower()


def test_game_streak_is_trivial():
    d = analyze(CHESS)
    assert d["triage"]["topic"] == "social_apps" and d["triage"]["category"] == "trivial"
    assert "AI/Can delete" in d["triage"]["gmail_labels"]
    assert d["draft"] is None


def test_bank_statement_is_important_no_reply():
    d = analyze(SBI, has_attachment=True)
    t = d["triage"]
    assert t["topic"] == "finance" and t["category"] == "important" and t["gmail_labels"] == ["AI/Finance"]
    assert t["needs_reply"] == "no" and "not to reply" in t["reply_reason"]
    assert {"label": "Attachment", "value": "Yes"} in t["key_details"]


def test_colleague_request_is_very_important_with_draft():
    d = analyze(COLLEAGUE)
    t = d["triage"]
    assert t["category"] == "very_important" and t["needs_reply"] == "yes" and t["priority"]
    assert t["topic"] == "work_academic" and t["gmail_labels"] == ["AI/Work and Academic", "AI/Priority"]
    assert t["deadline"] and "wednesday" in t["deadline"].lower()
    assert d["draft"] is not None


def test_force_draft_overrides_no_reply():
    assert analyze(SBI, force_draft=True)["draft"] is not None


def test_bulk_triage_and_sender_rule_correction():
    emails = [dict(e, id=str(i)) for i, e in enumerate([AMAZON, TLAUNCHER, CHESS, SBI, COLLEAGUE])]
    res = client.post("/triage", headers=H, json={"user_id": "bulk", "emails": emails}).json()["results"]
    topics = {r["id"]: r["topic"] for r in res}
    assert topics == {"0": "shopping_orders", "1": "accounts_security", "2": "social_apps", "3": "finance",
                      "4": "work_academic"}
    deletable = {r["id"] for r in res if r["can_delete"]}
    assert deletable == {"0", "2"}
    assert all(all(l.startswith("AI/") for l in r["gmail_labels"]) for r in res)

    # user: SBI alerts are Finance but always disposable, for the whole domain
    c = client.post("/triage/correct", headers=H, json={"user_id": "bulk", "sender": SBI["from"],
                    "predicted": "finance", "topic": "finance", "can_delete": True, "whole_domain": True}).json()
    assert c["sender_key"] == "@alerts.sbi.bank.in" and c["gmail_labels"] == ["AI/Finance", "AI/Can delete"]
    r2 = client.post("/triage", headers=H, json={"user_id": "bulk", "emails": [emails[3]]}).json()["results"][0]
    assert r2["topic"] == "finance" and r2["can_delete"] and "rule" in r2["method"]

    # user: chess.com is really Personal and should be kept
    client.post("/triage/correct", headers=H, json={"user_id": "bulk", "sender": CHESS["from"],
                "predicted": "social_apps", "topic": "personal", "can_delete": False})
    r3 = client.post("/triage", headers=H, json={"user_id": "bulk", "emails": [emails[2]]}).json()["results"][0]
    assert r3["topic"] == "personal" and not r3["can_delete"] and r3["gmail_labels"] == ["AI/Personal"]

    st = client.get("/stats", headers=H, params={"user_id": "bulk"}).json()["triage"]
    assert st["corrections"] == 2 and st["topic_corrections"] == 1 and st["emails_triaged"] == 7
    assert st["by_topic"]["finance"] == 2


def test_personal_email_from_a_friend():
    friend = {"from": "Anita Desai <anita.d@gmail.com>", "subject": "Dinner on Saturday?",
              "body": "Hi! We are planning dinner at our place on Saturday. Would you and the family like to come?"}
    t = analyze(friend)["triage"]
    assert t["topic"] == "personal" and t["needs_reply"] == "yes"


def test_promotion_is_advertisement():
    promo = {"from": "Myntra <offers@myntra.com>", "subject": "Flat 70% off ends tonight!",
             "body": "Biggest sale of the season. Shop now and get flat 70% off. Unsubscribe"}
    t = analyze(promo)["triage"]
    assert t["topic"] == "advertisements" and t["can_delete"]


def test_llm_triage_batch_is_redacted_and_merged(monkeypatch):
    from app import llm
    from app.config import settings
    seen = []

    def fake_json(system, user, **kw):
        seen.append(user)
        return {"results": [{"id": "0", "topic": "work_academic", "category": "very_important", "email_type": "work_academic",
                             "gist": "[PERSON_1] asks to present the DA2 demo on Thursday 3 pm.",
                             "sender_wants": "Confirm the Thursday 3 pm slot.", "needs_reply": "yes",
                             "action_items": ["Confirm the slot with [PERSON_1]"], "deadline": "Wednesday",
                             "key_details": [{"label": "Date", "value": "Thursday 3 pm"}],
                             "intent": "meeting_scheduling", "intent_confidence": 0.9, "phishing_risk": 0.0}]}

    monkeypatch.setattr(settings, "llm_api_key", "x")
    monkeypatch.setattr(llm, "complete_json", fake_json)
    res = client.post("/triage", headers=H, json={"user_id": "l", "emails": [dict(COLLEAGUE, id="0")]}).json()
    r = res["results"][0]
    assert "Rohan" not in seen[0] and "rohan.mehta@" not in seen[0] and "college.edu" in seen[0]
    assert r["gist"].startswith("Rohan") and r["action_items"] == ["Confirm the slot with Rohan Mehta"]
    assert r["method"].startswith("llm") and r["gmail_labels"] == ["AI/Work and Academic", "AI/Priority"]


def test_llm_failure_keeps_heuristic(monkeypatch):
    from app import llm
    from app.config import settings

    def boom(*a, **kw):
        raise llm.LLMError("429 quota")

    monkeypatch.setattr(settings, "llm_api_key", "x")
    monkeypatch.setattr(llm, "complete_json", boom)
    r = client.post("/triage", headers=H, json={"user_id": "f", "emails": [dict(CHESS, id="0")]}).json()["results"][0]
    assert r["topic"] == "social_apps" and "llm failed" in r["method"]
