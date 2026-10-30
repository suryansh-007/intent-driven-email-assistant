/**
 * Api.gs: talks to the Python backend.
 *
 * Set these in Apps Script → Project Settings → Script Properties:
 *   BACKEND_URL      e.g. https://email-assistant.onrender.com   (no trailing slash)
 *   BACKEND_API_KEY  the same value as BACKEND_API_KEY in the backend's .env
 *   USER_ID_SALT     any random string (keeps user IDs unlinkable to addresses)
 */

var MAX_TURNS = 10;          // newest N messages of a thread are sent
var MAX_BODY_CHARS = 20000;  // per message (raw HTML is bulky; the backend trims after cleaning)
var CACHE_SECONDS = 600;

function getConfig_() {
  var props = PropertiesService.getScriptProperties();
  var url = (props.getProperty('BACKEND_URL') || '').replace(/\/+$/, '');
  var key = props.getProperty('BACKEND_API_KEY') || '';
  if (!url || !key) {
    throw new Error('Add-on not configured: set BACKEND_URL and BACKEND_API_KEY in Script Properties.');
  }
  return { url: url, key: key, salt: props.getProperty('USER_ID_SALT') || 'intent-email-assistant' };
}

/** Pseudonymous, stable user id: SHA-256(salt + email). The raw address never leaves Google. */
function getUserId_() {
  var email = Session.getActiveUser().getEmail() || Session.getEffectiveUser().getEmail() || 'unknown';
  var bytes = Utilities.computeDigest(Utilities.DigestAlgorithm.SHA_256, getConfig_().salt + email.toLowerCase());
  return 'u_' + bytes.map(function (b) { return ('0' + (b & 0xff).toString(16)).slice(-2); }).join('').slice(0, 32);
}

function callBackend_(method, path, payload) {
  var cfg = getConfig_();
  var options = {
    method: method,
    contentType: 'application/json',
    headers: { 'X-API-Key': cfg.key },
    muteHttpExceptions: true
  };
  if (payload) options.payload = JSON.stringify(payload);
  var res;
  try {
    res = UrlFetchApp.fetch(cfg.url + path, options);
  } catch (err) {
    var msg = String(err && err.message || err);
    if (/whitelist|allowlist|not allowed/i.test(msg)) {
      throw new Error('Backend URL is not in "urlFetchWhitelist" in appsscript.json: ' + cfg.url);
    }
    throw new Error('Could not reach the backend (' + cfg.url + '). It may be asleep or down: ' + msg.slice(0, 150));
  }
  var code = res.getResponseCode();
  var text = res.getContentText();
  if (code === 401) throw new Error('Backend rejected the API key. Check BACKEND_API_KEY on both sides.');
  if (code === 530 || code === 502 || /error code: 10(16|33)/.test(text)) {
    throw new Error('The tunnel at ' + cfg.url + ' is not running. Restart cloudflared, then put the new ' +
                    'address in Script Properties (BACKEND_URL) and in urlFetchWhitelist.');
  }
  if (code >= 300) throw new Error('Backend error ' + code + ': ' + text.slice(0, 300));
  return JSON.parse(text);
}

/** Reads the open thread (oldest first) in the shape the backend expects. */
function readThread_(e) {
  GmailApp.setCurrentMessageAccessToken(e.gmail.accessToken);
  var message = GmailApp.getMessageById(e.gmail.messageId);
  var thread = message.getThread();
  var all = thread.getMessages();
  var msgs = all.slice(Math.max(0, all.length - MAX_TURNS)).map(function (m) {
    return {
      from: m.getFrom(),
      date: Utilities.formatDate(m.getDate(), Session.getScriptTimeZone(), 'EEE d MMM, HH:mm'),
      body: prepBody_(m, MAX_BODY_CHARS)   // plain text, or cleaned HTML (see Sort.gs)
    };
  });
  var last = all[all.length - 1];
  return { subject: message.getSubject() || thread.getFirstMessageSubject() || '', messages: msgs,
           lastMessageId: last.getId(),
           hasAttachment: last.getAttachments({ includeInlineImages: false }).length > 0 };
}

/** Runs the pipeline, with a short per-user cache so reopening a card is instant. */
function analyze_(e, opts) {
  opts = opts || {};
  var cache = CacheService.getUserCache();
  var cacheKey = 'an_' + e.gmail.messageId + '_' + (opts.tone || '') + '_' + (opts.force ? 1 : 0);
  if (!opts.guidance && !opts.noCache) {
    var hit = cache.get(cacheKey);
    if (hit) return JSON.parse(hit);
  }
  var thread = readThread_(e);
  var result = callBackend_('post', '/analyze', {
    user_id: getUserId_(),
    subject: thread.subject,
    messages: thread.messages,
    tone: opts.tone || null,
    guidance: opts.guidance || '',
    force_draft: !!opts.force,
    has_attachment: thread.hasAttachment
  });
  result._lastMessageId = thread.lastMessageId;
  var json = JSON.stringify(result);
  if (json.length < 90000) {
    cache.put(cacheKey, json, CACHE_SECONDS);
    if (result.draft) cache.put('draft_' + result.request_id, result.draft.text, CACHE_SECONDS * 3);
  }
  return result;
}

/** Feedback must never block the user, so failures are only logged. */
function sendFeedback_(payload) {
  try {
    payload.user_id = getUserId_();
    return callBackend_('post', '/feedback', payload);
  } catch (err) {
    console.warn('Feedback not recorded: ' + err);
    return null;
  }
}
