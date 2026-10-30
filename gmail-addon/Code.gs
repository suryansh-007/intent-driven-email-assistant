/**
 * Code.gs: Gmail add-on entry points, the email card, reply drafting and settings.
 *
 * Opening an email shows: its topic label, Priority / Can delete flags, what it is, what the
 * sender wants, key details, and whether it needs a reply; a draft only when a reply makes sense.
 * The homepage (no email open) sorts the inbox into AI/… labels; see Sort.gs.
 */

var INTENT_NAMES = {
  request_action: 'Action requested',
  question: 'Question',
  meeting_scheduling: 'Meeting / scheduling',
  approval_request: 'Approval requested',
  follow_up: 'Follow-up / reminder',
  complaint: 'Complaint / issue',
  information_fyi: 'FYI / update',
  social_thanks: 'Thanks / social',
  newsletter_promotion: 'Newsletter / promotion',
  other: 'Other'
};

var TONE_NAMES = { professional: 'Professional', friendly: 'Friendly', brief: 'Brief', formal: 'Formal' };
var REPLY_TEXT = { yes: 'Needs a reply', optional: 'Reply optional', no: 'No reply needed' };

// ---------------------------------------------------------------- homepage

function onHomepage(e) {
  return buildHomeCard_();
}

function onSortMenu(e) {
  return CardService.newUniversalActionResponseBuilder().displayAddOnCards([buildHomeCard_()]).build();
}

function buildHomeCard_() {
  var days = getUserProp_('sortDays', '30');
  var range = CardService.newSelectionInput().setType(CardService.SelectionInputType.DROPDOWN)
    .setTitle('Which emails').setFieldName('sort_days');
  [['7', 'Inbox, last 7 days'], ['30', 'Inbox, last 30 days'], ['90', 'Inbox, last 90 days'],
   ['0', 'Whole inbox']].forEach(function (o) { range.addItem(o[1], o[0], o[0] === days); });

  var sort = CardService.newCardSection().setHeader('Sort my inbox')
    .addWidget(CardService.newTextParagraph().setText(
      'Labels each conversation by topic, and marks what needs attention (Priority) and what is safe to ' +
      'remove (Can delete). Nothing is deleted unless you confirm.'))
    .addWidget(range)
    .addWidget(CardService.newTextButton().setText('Sort now')
      .setTextButtonStyle(CardService.TextButtonStyle.FILLED)
      .setOnClickAction(CardService.newAction().setFunctionName('onSortInbox')));

  var labels = CardService.newCardSection().setHeader('Labels').setCollapsible(true)
    .setNumUncollapsibleWidgets(2);
  [PRIORITY_LABEL, DELETE_LABEL].concat(TOPIC_ORDER.map(function (t) { return TOPICS[t].label; }))
    .forEach(function (name) {
      labels.addWidget(CardService.newDecoratedText().setText(escape_(name)).setButton(openButton_(name)));
    });

  var clean = CardService.newCardSection().setHeader('Clean up')
    .addWidget(CardService.newTextButton().setText('Delete in bulk')
      .setOnClickAction(CardService.newAction().setFunctionName('onCleanupMenu')));

  var me = CardService.newCardSection();
  try {
    var s = callBackend_('get', '/settings/' + getUserId_());
    var c = s.counts || {};
    var total = (c.accept || 0) + (c.edit || 0) + (c.reject || 0);
    me.addWidget(CardService.newDecoratedText().setTopLabel('Reply drafts so far')
      .setText(total ? (c.accept + ' accepted, ' + c.edit + ' edited, ' + c.reject + ' rejected') : 'None used yet')
      .setWrapText(true));
  } catch (err) {
    me.addWidget(CardService.newTextParagraph().setText('<font color="#b3261e">' + escape_(err.message) + '</font>'));
  }
  me.addWidget(CardService.newTextButton().setText('Settings')
    .setOnClickAction(CardService.newAction().setFunctionName('onOpenSettings').setParameters({ fromButton: '1' })));

  return CardService.newCardBuilder()
    .setHeader(CardService.newCardHeader().setTitle('Intent Email Assistant')
      .setSubtitle('Open an email for its summary and a reply, or sort your inbox here'))
    .addSection(sort).addSection(clean).addSection(labels).addSection(me)
    .build();
}

// ---------------------------------------------------------------- email opened

function onGmailMessageOpen(e) {
  if (getUserProp_('autoAnalyze', 'true') !== 'true') return buildIdleCard_();
  return analyzeAndBuild_(e, {});
}

function onAnalyzeClicked(e) { return navUpdate_(analyzeAndBuild_(e, {})); }

function onRegenerate(e) {
  return navUpdate_(analyzeAndBuild_(e, {
    tone: formValue_(e, 'tone') || null,
    guidance: (formValue_(e, 'guidance') || '').slice(0, 500),
    force: e.parameters && e.parameters.force === '1',
    noCache: true
  }));
}

function onDraftAnyway(e) { return navUpdate_(analyzeAndBuild_(e, { force: true, noCache: true })); }

function analyzeAndBuild_(e, opts) {
  try {
    var r = analyze_(e, opts);
    if (getUserProp_('autoLabel', 'true') === 'true' && r.triage) {
      try {
        GmailApp.setCurrentMessageAccessToken(e.gmail.accessToken);
        setThreadLabels_(GmailApp.getMessageById(e.gmail.messageId).getThread(), r.triage.gmail_labels || []);
      } catch (err) { console.warn('Auto-label failed: ' + err); }
    }
    return buildResultCard_(r, opts);
  } catch (err) {
    console.error(err);
    return buildErrorCard_(err.message);
  }
}

function buildResultCard_(r, opts) {
  var t = r.triage || {};
  var topicName = (TOPICS[t.topic] || TOPICS.personal).name;
  var card = CardService.newCardBuilder();
  var tags = [t.importance_name || '', REPLY_TEXT[t.needs_reply] || ''].filter(String).join(' · ');
  card.setHeader(CardService.newCardHeader()
    .setTitle(topicName + (t.priority ? ' · Priority' : '') + (t.can_delete ? ' · Can delete' : ''))
    .setSubtitle(tags + (r.n_turns > 1 ? ' · ' + r.n_turns + ' messages' : '')));

  // Security warning first
  if ((r.phishing && r.phishing.suspect) || t.topic === 'spam') {
    var warn = CardService.newCardSection().setHeader('Security warning')
      .addWidget(CardService.newTextParagraph().setText(
        '<font color="#b3261e"><b>This looks like spam or phishing.</b> Do not open its links or share passwords or codes.</font>'));
    if (r.phishing && r.phishing.reasons && r.phishing.reasons.length) {
      warn.addWidget(CardService.newTextParagraph().setText('Signals: ' + escape_(r.phishing.reasons.join('; '))));
    }
    card.addSection(warn);
  }

  // What is it / what do they want
  var about = CardService.newCardSection().setHeader('Summary')
    .addWidget(CardService.newTextParagraph().setText('<b>' + escape_(t.gist || '') + '</b>'))
    .addWidget(CardService.newDecoratedText().setTopLabel('What the sender wants')
      .setText(escape_(t.sender_wants || '')).setWrapText(true));
  (t.key_details || []).forEach(function (d) {
    about.addWidget(CardService.newDecoratedText().setTopLabel(escape_(d.label)).setText(escape_(d.value)).setWrapText(true));
  });
  if (t.deadline) {
    about.addWidget(CardService.newDecoratedText().setTopLabel('Deadline')
      .setText('<font color="#b3261e"><b>' + escape_(t.deadline) + '</b></font>').setWrapText(true));
  }
  if (t.action_items && t.action_items.length) {
    about.addWidget(CardService.newDecoratedText().setTopLabel('Suggested actions')
      .setText(t.action_items.map(function (a) { return '- ' + escape_(a); }).join('<br>')).setWrapText(true));
  }
  about.addWidget(CardService.newDecoratedText().setTopLabel('Labels')
    .setText(escape_((t.gmail_labels || []).join(', '))).setBottomLabel(escape_(t.category_reason || ''))
    .setWrapText(true));
  card.addSection(about);

  // Thread summary (multi-message threads or long emails)
  if (r.summary && r.n_turns > 1) {
    var faith = Math.round(r.summary.faithfulness * 100);
    var colour = faith >= 70 ? '#188038' : (faith >= 50 ? '#b06000' : '#b3261e');
    card.addSection(CardService.newCardSection().setHeader('Thread summary')
      .addWidget(CardService.newTextParagraph().setText(escape_(r.summary.text)))
      .addWidget(CardService.newTextParagraph().setText('<font color="' + colour + '">Faithfulness ' + faith + '%</font>' +
        '<font color="#5f6368"> · ' + escape_(r.summary.method) + (r.summary.fell_back ? ' (fallback)' : '') + '</font>')));
  }

  // Reply
  var params = { requestId: r.request_id, intent: r.intent.label, lastMessageId: r._lastMessageId || '' };
  var reply = CardService.newCardSection().setHeader('Reply');
  if (r.draft) {
    reply.addWidget(CardService.newTextParagraph().setText('<font color="#5f6368">' +
      escape_(REPLY_TEXT[t.needs_reply] || '') + ': ' + escape_(t.reply_reason || '') + '</font>'));
    reply.addWidget(CardService.newTextInput().setFieldName('draft').setTitle('Draft (edit before inserting)')
      .setMultiline(true).setValue(r.draft.text));
    reply.addWidget(CardService.newSelectionInput().setType(CardService.SelectionInputType.CHECK_BOX)
      .setFieldName('reply_all').addItem('Reply all', 'yes', false));
    reply.addWidget(CardService.newButtonSet()
      .addButton(CardService.newTextButton().setText('Insert as reply')
        .setTextButtonStyle(CardService.TextButtonStyle.FILLED)
        .setComposeAction(CardService.newAction().setFunctionName('onInsertDraft').setParameters(params),
          CardService.ComposedEmailType.REPLY_AS_DRAFT))
      .addButton(CardService.newTextButton().setText('Not useful')
        .setOnClickAction(CardService.newAction().setFunctionName('onReject').setParameters(params))));
  } else {
    var why = r.draft_skipped === 'phishing_suspect' ? 'Don\'t reply to suspected spam or phishing.'
      : (t.reply_reason || 'No reply needed.');
    reply.addWidget(CardService.newDecoratedText().setText('<b>No reply needed</b>').setBottomLabel(why).setWrapText(true));
    reply.addWidget(CardService.newTextButton().setText('Draft a reply anyway')
      .setOnClickAction(CardService.newAction().setFunctionName('onDraftAnyway')));
  }
  card.addSection(reply);

  if (r.draft) {
    var tone = CardService.newSelectionInput().setType(CardService.SelectionInputType.DROPDOWN)
      .setTitle('Tone').setFieldName('tone');
    (r.tones || Object.keys(TONE_NAMES)).forEach(function (k) { tone.addItem(TONE_NAMES[k] || k, k, k === r.tone); });
    card.addSection(CardService.newCardSection().setHeader('Steer the reply').setCollapsible(true)
      .setNumUncollapsibleWidgets(0)
      .addWidget(tone)
      .addWidget(CardService.newTextInput().setFieldName('guidance').setTitle('Instruction (optional)')
        .setHint('e.g. "accept, but propose Friday 4 pm instead"').setValue((opts && opts.guidance) || ''))
      .addWidget(CardService.newTextButton().setText('Regenerate')
        .setOnClickAction(CardService.newAction().setFunctionName('onRegenerate').setParameters({ force: '1' }))));
  }

  // Organize: trash, wrong label
  var org = CardService.newCardSection().setHeader('Organize');
  var trashBtn = CardService.newTextButton().setText('Move to Trash')
    .setOnClickAction(CardService.newAction().setFunctionName('onTrashThis'));
  if (t.can_delete) trashBtn.setTextButtonStyle(CardService.TextButtonStyle.FILLED);
  org.addWidget(trashBtn);
  var pick = CardService.newSelectionInput().setType(CardService.SelectionInputType.DROPDOWN)
    .setTitle('Wrong label? Choose the right topic').setFieldName('correct_topic');
  TOPIC_ORDER.forEach(function (k) { pick.addItem(TOPICS[k].name, k, k === t.topic); });
  org.addWidget(pick);
  org.addWidget(CardService.newSelectionInput().setType(CardService.SelectionInputType.DROPDOWN)
    .setTitle('Mail like this can be deleted').setFieldName('correct_delete')
    .addItem('Let the assistant decide', 'auto', true)
    .addItem('Yes, always mark Can delete', 'yes', false)
    .addItem('No, never mark Can delete', 'no', false));
  var domain = (r.sender_key || '').split('@')[1] || '';
  if (domain) {
    org.addWidget(CardService.newSelectionInput().setType(CardService.SelectionInputType.CHECK_BOX)
      .setFieldName('whole_domain').addItem('Apply to all mail from @' + domain, 'yes', !!t.automated));
  }
  org.addWidget(CardService.newTextButton().setText('Save correction')
    .setOnClickAction(CardService.newAction().setFunctionName('onCorrectLabel')
      .setParameters({ predicted: t.topic || '', priority: t.priority ? '1' : '0' })));
  card.addSection(org);

  // Details for the demo / evaluation
  var top = (r.intent.top || []).map(function (x) {
    return (INTENT_NAMES[x[0]] || humanize_(x[0])) + ' ' + Math.round(x[1] * 100) + '%';
  }).join(' · ');
  card.addSection(CardService.newCardSection().setHeader('Pipeline details').setCollapsible(true)
    .setNumUncollapsibleWidgets(0)
    .addWidget(CardService.newTextParagraph().setText(
      'Intent: ' + escape_(r.intent.is_open ? 'New intent: ' + humanize_(r.intent.action_object || r.intent.label)
                                             : (INTENT_NAMES[r.intent.label] || humanize_(r.intent.label))) +
      ' (' + Math.round(r.intent.confidence * 100) + '%, ' + escape_(r.intent.method) + ')' +
      (top ? '<br>Top intents: ' + escape_(top) : '') +
      '<br>Triage: ' + escape_(t.method || '') +
      (r.draft ? '<br>Generator: ' + escape_(r.draft.method) : '') +
      '<br>Latency: ' + escape_(Object.keys(r.latency_ms).map(function (k) { return k + ' ' + Math.round(r.latency_ms[k]) + ' ms'; }).join(', ')) +
      '<br>PII items redacted: ' + r.redactions +
      (r.kb_sources && r.kb_sources.length ? '<br>Knowledge base: ' + escape_(r.kb_sources.join(', ')) : '') +
      '<br>Request: ' + r.request_id)));

  return card.build();
}

// ---------------------------------------------------------------- reply actions

/** Compose action: opens a reply draft with the (possibly edited) text and logs feedback. */
function onInsertDraft(e) {
  var p = e.parameters || {};
  var finalText = formValue_(e, 'draft') || '';
  var original = CacheService.getUserCache().get('draft_' + p.requestId) || '';
  var action = finalText.trim() === original.trim() ? 'accept' : 'edit';
  sendFeedback_({ request_id: p.requestId, action: action, draft: original, final_text: finalText, intent: p.intent || '' });

  GmailApp.setCurrentMessageAccessToken(e.gmail.accessToken);
  var msg = GmailApp.getMessageById(p.lastMessageId || e.gmail.messageId);
  var draft = formValue_(e, 'reply_all') === 'yes' ? msg.createDraftReplyAll(finalText) : msg.createDraftReply(finalText);
  return CardService.newComposeActionResponseBuilder().setGmailDraft(draft).build();
}

function onReject(e) {
  var p = e.parameters || {};
  var original = CacheService.getUserCache().get('draft_' + p.requestId) || '';
  sendFeedback_({ request_id: p.requestId, action: 'reject', draft: original, intent: p.intent || '' });
  return notify_('Thanks. Noted. Try another tone or add an instruction, then Regenerate.');
}

// ---------------------------------------------------------------- settings

function onOpenSettings(e) {
  var s = {};
  try { s = callBackend_('get', '/settings/' + getUserId_()); } catch (err) { s = { error: err.message }; }

  var tone = CardService.newSelectionInput().setType(CardService.SelectionInputType.DROPDOWN)
    .setTitle('Default tone').setFieldName('tone');
  Object.keys(TONE_NAMES).forEach(function (k) { tone.addItem(TONE_NAMES[k], k, k === (s.tone || 'professional')); });

  var section = CardService.newCardSection()
    .addWidget(CardService.newTextInput().setFieldName('display_name').setTitle('Your name (for signatures)')
      .setValue(s.display_name || ''))
    .addWidget(CardService.newTextInput().setFieldName('signoff').setTitle('Sign-off')
      .setHint('e.g. "Best regards," (learned automatically from your edits)').setValue(s.signoff || ''))
    .addWidget(tone)
    .addWidget(CardService.newSelectionInput().setType(CardService.SelectionInputType.CHECK_BOX)
      .setFieldName('auto').addItem('Analyze emails automatically when opened', 'true',
        getUserProp_('autoAnalyze', 'true') === 'true'))
    .addWidget(CardService.newSelectionInput().setType(CardService.SelectionInputType.CHECK_BOX)
      .setFieldName('auto_label').addItem('Add the AI label to emails I open', 'true',
        getUserProp_('autoLabel', 'true') === 'true'))
    .addWidget(CardService.newSelectionInput().setType(CardService.SelectionInputType.CHECK_BOX)
      .setFieldName('auto_sort').addItem('Auto-sort new emails every hour', 'true', isAutoSortOn_()))
    .addWidget(CardService.newTextButton().setText('Save')
      .setTextButtonStyle(CardService.TextButtonStyle.FILLED)
      .setOnClickAction(CardService.newAction().setFunctionName('onSaveSettings')));
  if (s.error) section.addWidget(CardService.newTextParagraph().setText('<font color="#b3261e">' + escape_(s.error) + '</font>'));

  var card = CardService.newCardBuilder()
    .setHeader(CardService.newCardHeader().setTitle('Assistant settings'))
    .addSection(section).build();
  if (e && e.parameters && e.parameters.fromButton) return navPush_(card);
  return CardService.newUniversalActionResponseBuilder().displayAddOnCards([card]).build();
}

function onSaveSettings(e) {
  setUserProp_('autoAnalyze', formValue_(e, 'auto') === 'true' ? 'true' : 'false');
  setUserProp_('autoLabel', formValue_(e, 'auto_label') === 'true' ? 'true' : 'false');
  var notes = [];
  try {
    setAutoSort_(formValue_(e, 'auto_sort') === 'true');
  } catch (err) {
    notes.push('auto-sort could not be scheduled (' + err.message + ')');
  }
  try {
    callBackend_('post', '/settings', {
      user_id: getUserId_(),
      tone: formValue_(e, 'tone') || null,
      signoff: formValue_(e, 'signoff') || '',
      display_name: formValue_(e, 'display_name') || ''
    });
  } catch (err) {
    notes.push('could not save to backend: ' + err.message);
  }
  return notify_(notes.length ? 'Saved, but ' + notes.join('; ') : 'Settings saved');
}

// ---------------------------------------------------------------- small cards

function buildIdleCard_() {
  return CardService.newCardBuilder()
    .setHeader(CardService.newCardHeader().setTitle('Intent Email Assistant'))
    .addSection(CardService.newCardSection()
      .addWidget(CardService.newTextButton().setText('Analyze this email')
        .setTextButtonStyle(CardService.TextButtonStyle.FILLED)
        .setOnClickAction(CardService.newAction().setFunctionName('onAnalyzeClicked'))))
    .build();
}

function buildErrorCard_(message) {
  return CardService.newCardBuilder()
    .setHeader(CardService.newCardHeader().setTitle('Something went wrong'))
    .addSection(CardService.newCardSection()
      .addWidget(CardService.newTextParagraph().setText('<font color="#b3261e">' + escape_(message) + '</font>'))
      .addWidget(CardService.newTextButton().setText('Try again')
        .setOnClickAction(CardService.newAction().setFunctionName('onAnalyzeClicked')))
      .addWidget(CardService.newTextButton().setText('Settings')
        .setOnClickAction(CardService.newAction().setFunctionName('onOpenSettings').setParameters({ fromButton: '1' }))))
    .build();
}

// ---------------------------------------------------------------- helpers

function formValue_(e, name) {
  var f = e && e.commonEventObject && e.commonEventObject.formInputs && e.commonEventObject.formInputs[name];
  if (f && f.stringInputs && f.stringInputs.value && f.stringInputs.value.length) return f.stringInputs.value[0];
  if (e && e.formInput && e.formInput[name] !== undefined) return e.formInput[name];
  return '';
}

function navUpdate_(card) {
  return CardService.newActionResponseBuilder().setNavigation(CardService.newNavigation().updateCard(card)).build();
}

function navPush_(card) {
  return CardService.newActionResponseBuilder().setNavigation(CardService.newNavigation().pushCard(card)).build();
}

function notify_(text) {
  return CardService.newActionResponseBuilder().setNotification(CardService.newNotification().setText(text)).build();
}

function getUserProp_(k, dflt) {
  var v = PropertiesService.getUserProperties().getProperty(k);
  return v === null || v === undefined ? dflt : v;
}

function setUserProp_(k, v) { PropertiesService.getUserProperties().setProperty(k, v); }

function humanize_(s) {
  s = String(s || '').replace(/^open:/, '').replace(/_/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function escape_(s) {
  return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
