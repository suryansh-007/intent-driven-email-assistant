# Gmail add-on: setup

The add-on is a thin client. It reads the open thread, sends it to the backend
(`../backend`), and shows the result in Gmail's right-hand side panel.

**When an email is open** it shows the email's topic (Personal, Work & Academic, Finance,
Shopping & Orders, Advertisements, Social & Apps, Newsletters & Blogs, Accounts & Security, Spam),
whether it is Priority or Can delete, what the email is, what the sender wants, key details
(order numbers, amounts, dates, deadlines), whether it needs a reply, and a draft only
when a reply makes sense. "Wrong label?" teaches it a rule for that sender or domain.

**With no email open** (or ⋮ → *Sort my inbox*) it sorts the inbox into Gmail labels
(`AI/Personal`, `AI/Work and Academic`, `AI/Finance`, `AI/Shopping and Orders`, `AI/Advertisements`, `AI/Social and Apps`, `AI/Newsletters and Blogs`, `AI/Accounts and Security`, `AI/Spam`, plus `AI/Priority` and `AI/Can delete`), and offers **Clean up**, which moves
them to Trash (recoverable for 30 days; starred mail is never touched). Settings can turn on
**hourly auto-sort**.

```
Gmail side panel                         Backend (FastAPI)
────────────────                         ─────────────────
open email ──► Api.gs: readThread_ ──►  POST /analyze  → redact → intent → summary
                                                         → faithfulness → persona + KB → draft
card: intent · summary · draft  ◄──────  JSON
Insert as reply ──► Gmail reply draft
                └─► POST /feedback (accept / edit)  → persona store + evaluation log
Not useful ───────► POST /feedback (reject)
```

## 1. Get the backend reachable over HTTPS

Apps Script can only call public **https** URLs. Pick one option:

| Option | Good for | Notes |
|---|---|---|
| **Tunnel to your laptop** (`cloudflared tunnel --url http://localhost:8000` or `ngrok http 8000`) | development, demos | Free and instant. The URL changes each time you restart the tunnel, so update step 4 when it does. |
| **Google Cloud Run** (`gcloud run deploy --source backend`) | DA2/DA3 demo, user study | Starts fast, has a free tier, needs a billing account. |
| **Hugging Face Spaces** (Docker Space) | free hosting | Free CPU. The Space sleeps after inactivity. |
| **Render / Railway** | free hosting | The free tier sleeps after about 15 minutes. The first request then takes 30–60 s, which is longer than Gmail's 30-second card limit, so open the `/health` URL once to wake it before a demo. |

Check that it works: `https://<your-url>/health` should return `{"status":"ok",…}`.

## 2. Create the Apps Script project

**Option A: in the browser**
1. Go to <https://script.google.com>, then **New project**, and rename it to "Intent Email Assistant".
2. **Project Settings** (gear icon): tick **Show "appsscript.json" manifest file in editor**.
3. In the editor, replace the contents of `appsscript.json` with this folder's file. Create
   `Api.gs`, `Sort.gs` and `Code.gs` (click **+**, then **Script**) and paste in the code. Replace the default `Code.gs` content.

**Option B: with clasp (command line)**
```bash
npm install -g @google/clasp
clasp login
cd gmail-addon
clasp create --type standalone --title "Intent Email Assistant"
clasp push
```

## 3. Script Properties

**Project Settings, then Script Properties, then Add property:**

| Property | Value |
|---|---|
| `BACKEND_URL` | `https://<your-url>` (no trailing slash) |
| `BACKEND_API_KEY` | the same value as `BACKEND_API_KEY` in `backend/.env` |
| `USER_ID_SALT` | any random string; it keeps user IDs pseudonymous |

## 4. Allowlist the backend URL

In `appsscript.json`, change `urlFetchWhitelist` to your backend URL, ending with `/`:
```json
"urlFetchWhitelist": ["https://abc-123.trycloudflare.com/"]
```

## 5. Install it in your Gmail

1. **Deploy, then Test deployments**. Under **Application(s): Gmail**, click **Install**.
2. Open or refresh Gmail. A new icon appears in the right-hand side panel. Open any email and click it.
3. The first time, Google asks you to authorize. Because the app is unverified, you'll see "Google hasn't verified this app": click **Advanced**, then **Go to Intent Email Assistant**. This is expected for a course project.

To let teammates or study participants use it, share the Apps Script project with them (Editor access). Each of them then does step 5. A test deployment works for up to about 100 test users without Google's verification review.

## Permissions it asks for

| Scope | Why |
|---|---|
| `gmail.addons.execute` | run as a Gmail add-on |
| `gmail.addons.current.message.readonly` | read the email you have open |
| `gmail.addons.current.action.compose` | create the reply draft when you click *Insert as reply* |
| `https://mail.google.com/` | **Sort my inbox**: search the inbox, add AI labels, move groups you choose to Trash. This is Google's broad Gmail scope: fine for a test deployment, but publishing publicly would need Google's security review |
| `script.scriptapp` | the optional hourly auto-sort trigger |
| `script.external_request` | call the backend |
| `userinfo.email` | derive your pseudonymous user ID (the address itself is hashed and never sent) |

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Add-on not configured" | Script Properties are missing (step 3). |
| "not in urlFetchWhitelist" | Step 4. The URL must match exactly, including `https://` and the trailing `/`. |
| "Backend rejected the API key" | `BACKEND_API_KEY` differs between Script Properties and `backend/.env`. |
| Card times out / spins | The backend is asleep (Render or HF) or a local model is too slow on CPU. Wake it via `/health`, or use an LLM backend. |
| A college Google Workspace account can't install it | Your admin may block unverified apps. Use a personal Gmail account for development. |
| Logs | Apps Script editor, then **Executions**. |
| "Sort now" only sorts ~20–40 emails per click | Gmail add-on buttons must finish within 30 seconds. Click **Continue sorting**, or turn on hourly auto-sort in Settings (each run can take up to 6 minutes). |
| Categories look shallow / everything says "heuristic" | The LLM isn't being used. Check `/health`: `llm_provider` must not be `null`. The **Pipeline details** section shows `llm failed: …` with the reason. |
| After updating the code, it asks for permission again | Expected: new features need new permissions. Click through as in step 5. |

## Upgrading from the earlier version (AI/1 Very important … AI/5 Spam)

The old priority labels are no longer used. To tidy up, in Gmail's left sidebar open each old label
(`AI/1 Very important`, `AI/2 Important`, `AI/3 FYI`, `AI/4 Trivial`, `AI/5 Spam`), click the
three dots next to it and choose **Remove label**. Your emails are not deleted. Then run
**Sort my inbox** again with "Whole inbox" so every conversation gets its topic label.
