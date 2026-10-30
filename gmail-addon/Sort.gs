/**
 * Sort.gs: inbox sorting into topic labels, bulk delete, and hourly auto-sort.
 *
 * Every conversation gets exactly one topic label (nested under "AI" in Gmail's sidebar):
 *   AI/Personal · AI/Work and Academic · AI/Finance · AI/Shopping and Orders · AI/Advertisements ·
 *   AI/Social and Apps · AI/Newsletters and Blogs · AI/Accounts and Security · AI/Spam
 * (label names spell out "and" because Gmail search handles "&" in label names unreliably)
 * plus, when relevant:
 *   AI/Priority    needs your attention soon
 *   AI/Can delete  safe to delete without reading
 * Sorting only adds labels. Nothing is deleted until you confirm, deleted mail goes to Trash
 * (recoverable for 30 days), and starred mail is never deleted.
 */

var TOPICS = {
  personal:          { name: 'Personal',            label: 'AI/Personal' },
  work_academic:     { name: 'Work & Academic',     label: 'AI/Work and Academic' },
  finance:           { name: 'Finance',             label: 'AI/Finance' },
  shopping_orders:   { name: 'Shopping & Orders',   label: 'AI/Shopping and Orders' },
  advertisements:    { name: 'Advertisements',      label: 'AI/Advertisements' },
  social_apps:       { name: 'Social & Apps',       label: 'AI/Social and Apps' },
  newsletters_blogs: { name: 'Newsletters & Blogs', label: 'AI/Newsletters and Blogs' },
  accounts_security: { name: 'Accounts & Security', label: 'AI/Accounts and Security' },
  spam:              { name: 'Spam',                label: 'AI/Spam' }
};
var TOPIC_ORDER = ['personal', 'work_academic', 'finance', 'shopping_orders', 'advertisements',
                   'social_apps', 'newsletters_blogs', 'accounts_security', 'spam'];
var PRIORITY_LABEL = 'AI/Priority';
var DELETE_LABEL = 'AI/Can delete';
// labels offered for one-click bulk delete
var DELETABLE = [
  { key: 'can_delete', name: 'Can delete', label: DELETE_LABEL },
  { key: 'advertisements', name: 'Advertisements', label: TOPICS.advertisements.label },
  { key: 'social_apps', name: 'Social & Apps', label: TOPICS.social_apps.label },
  { key: 'spam', name: 'Spam', label: TOPICS.spam.label }
];

var SORT_BATCH = 10;            // threads per backend call
var CARD_BUDGET_MS = 22000;     // card actions must finish within ~30 s
var TRIGGER_BUDGET_MS = 270000; // time-driven runs may take up to 6 min

// ---------------------------------------------------------------- labels

var labelCache_ = {};

function getLabel_(name) {
  if (labelCache_[name]) return labelCache_[name];
  var label = GmailApp.getUserLabelByName(name);
  if (!label) {
    if (!GmailApp.getUserLabelByName('AI')) GmailApp.createLabel('AI');
    label = GmailApp.createLabel(name);
  }
  labelCache_[name] = label;
  return label;
}

/** Gmail search syntax for a label: "AI/Work and Academic" -> "label:ai-work-and-academic" */
function labelQuery_(name) {
  return 'label:' + name.toLowerCase().replace(/[\s\/]+/g, '-');
}

function labelUrl_(name) {
  return 'https://mail.google.com/mail/u/0/#label/' + encodeURIComponent(name);
}

function allAiLabels_() {
  return TOPIC_ORDER.map(function (t) { return TOPICS[t].label; }).concat([PRIORITY_LABEL, DELETE_LABEL]);
}

/** Sets a thread's AI labels to exactly `labels` (removes any other AI/… labels). */
function setThreadLabels_(thread, labels) {
  allAiLabels_().forEach(function (name) {
    if (labels.indexOf(name) === -1) {
      var l = GmailApp.getUserLabelByName(name);
      if (l) thread.removeLabel(l);
    }
  });
  labels.forEach(function (name) { thread.addLabel(getLabel_(name)); });
}

function unsortedQuery_(days) {
  var q = 'in:inbox' + (days ? ' newer_than:' + days + 'd' : '');
  TOPIC_ORDER.forEach(function (t) { q += ' -' + labelQuery_(TOPICS[t].label); });
  return q;
}

/** Plain text if available; otherwise HTML with <head>/<style>/<script> removed (the backend converts the rest). */
function prepBody_(message, maxChars) {
  var body = message.getPlainBody();
  if (!body || /<html|<!doctype/i.test(body.slice(0, 500))) {
    body = (message.getBody() || '')
      .replace(/<head[\s\S]*?<\/head>/i, ' ')
      .replace(/<(style|script)[\s\S]*?<\/\1>/gi, ' ')
      .replace(/<!--[\s\S]*?-->/g, ' ');
  }
  return (body || '').slice(0, maxChars);
}

// ---------------------------------------------------------------- sorting

/** Labels unsorted inbox threads until the time budget runs out. */
function sortInbox_(days, budgetMs) {
  var start = Date.now();
  var counts = {}, samples = {};
  TOPIC_ORDER.forEach(function (t) { counts[t] = 0; samples[t] = []; });
  var nPriority = 0, nDelete = 0, processed = 0, seen = {}, error = null, lastBatchMs = 0;
  var query = unsortedQuery_(days);

  while (Date.now() - start + lastBatchMs * 1.2 < budgetMs) {
    var t0 = Date.now();
    var threads = GmailApp.search(query, 0, SORT_BATCH).filter(function (t) { return !seen[t.getId()]; });
    if (!threads.length) break;
    var emails = threads.map(function (t) {
      seen[t.getId()] = true;
      var msgs = t.getMessages();
      var last = msgs[msgs.length - 1];
      return {
        id: t.getId(),
        from: last.getFrom(),
        subject: t.getFirstMessageSubject() || '',
        body: prepBody_(last, 6000),
        has_attachment: last.getAttachments({ includeInlineImages: false }).length > 0
      };
    });
    var res;
    try {
      res = callBackend_('post', '/triage', { user_id: getUserId_(), emails: emails }).results;
    } catch (err) {
      error = err.message;
      break;
    }
    var byId = {};
    threads.forEach(function (t) { byId[t.getId()] = t; });
    var groups = {};   // label name -> threads
    res.forEach(function (r) {
      var topic = TOPICS[r.topic] ? r.topic : 'personal';
      counts[topic]++;
      if (r.priority) nPriority++;
      if (r.can_delete) nDelete++;
      if (samples[topic].length < 3) samples[topic].push((r.gist || '').slice(0, 80));
      (r.gmail_labels || [TOPICS[topic].label]).forEach(function (name) {
        (groups[name] = groups[name] || []).push(byId[r.id]);
      });
    });
    Object.keys(groups).forEach(function (name) {
      var list = groups[name].filter(Boolean);
      for (var i = 0; i < list.length; i += 100) getLabel_(name).addToThreads(list.slice(i, i + 100));
    });
    processed += res.length;
    lastBatchMs = Date.now() - t0;
  }
  var remaining = error ? null : GmailApp.search(query, 0, 200).length;
  return { counts: counts, samples: samples, priority: nPriority, deletable: nDelete,
           processed: processed, remaining: remaining, error: error };
}

function onSortInbox(e) {
  var days = Number(formValue_(e, 'sort_days') || (e.parameters && e.parameters.days) || 30);
  setUserProp_('sortDays', String(days));
  return navPush_(buildSortResultCard_(sortInbox_(days, CARD_BUDGET_MS), days));
}

function openButton_(label) {
  return CardService.newTextButton().setText('Open')
    .setOpenLink(CardService.newOpenLink().setUrl(labelUrl_(label)));
}

function buildSortResultCard_(r, days) {
  var card = CardService.newCardBuilder().setHeader(CardService.newCardHeader()
    .setTitle(r.processed ? 'Sorted ' + r.processed + ' conversations' : 'Nothing new to sort')
    .setSubtitle(days ? 'Inbox, last ' + days + ' days' : 'Whole inbox'));

  if (r.error) {
    card.addSection(CardService.newCardSection().addWidget(CardService.newTextParagraph()
      .setText('<font color="#b3261e">' + escape_(r.error) + '</font>')));
  }

  var flags = CardService.newCardSection().setHeader('Attention')
    .addWidget(CardService.newDecoratedText().setText('Priority: <b>' + r.priority + '</b>')
      .setBottomLabel('Need your attention soon').setButton(openButton_(PRIORITY_LABEL)))
    .addWidget(CardService.newDecoratedText().setText('Can delete: <b>' + r.deletable + '</b>')
      .setBottomLabel('Safe to delete without reading').setButton(openButton_(DELETE_LABEL)));
  card.addSection(flags);

  var section = CardService.newCardSection().setHeader('By topic');
  TOPIC_ORDER.forEach(function (t) {
    if (!r.counts[t]) return;
    var w = CardService.newDecoratedText().setText(TOPICS[t].name + ': <b>' + r.counts[t] + '</b>')
      .setWrapText(true).setButton(openButton_(TOPICS[t].label));
    if (r.samples[t].length) w.setBottomLabel(r.samples[t].join(' · '));
    section.addWidget(w);
  });
  if (!r.processed) section.addWidget(CardService.newTextParagraph().setText('No unsorted conversations in this range.'));
  card.addSection(section);

  var next = CardService.newCardSection();
  if (r.remaining) {
    next.addWidget(CardService.newTextButton()
      .setText('Continue sorting (' + (r.remaining >= 200 ? '200+' : r.remaining) + ' left)')
      .setTextButtonStyle(CardService.TextButtonStyle.FILLED)
      .setOnClickAction(CardService.newAction().setFunctionName('onSortInbox').setParameters({ days: String(days) })));
  }
  next.addWidget(CardService.newTextButton().setText('Clean up (delete in bulk)')
    .setOnClickAction(CardService.newAction().setFunctionName('onCleanupMenu')));
  next.addWidget(CardService.newTextParagraph().setText(
    '<font color="#5f6368">Sorting only adds labels; find them under AI in Gmail\'s left sidebar. ' +
    'If a label is wrong, open the email and use "Wrong label?" to correct it.</font>'));
  card.addSection(next);
  return card.build();
}

// ---------------------------------------------------------------- bulk delete

function deletableByKey_(key) {
  for (var i = 0; i < DELETABLE.length; i++) if (DELETABLE[i].key === key) return DELETABLE[i];
  return null;
}

function deleteQuery_(label) { return labelQuery_(label) + ' -is:starred'; }

function onCleanupMenu(e) {
  var section = CardService.newCardSection()
    .addWidget(CardService.newTextParagraph().setText(
      'Move whole groups to Trash. You will see a preview first. Starred mail is always skipped.'));
  DELETABLE.forEach(function (d) {
    section.addWidget(CardService.newDecoratedText().setText(d.name).setBottomLabel(d.label)
      .setButton(CardService.newTextButton().setText('Review')
        .setOnClickAction(CardService.newAction().setFunctionName('onConfirmDelete').setParameters({ key: d.key }))));
  });
  return navPush_(CardService.newCardBuilder()
    .setHeader(CardService.newCardHeader().setTitle('Clean up')).addSection(section).build());
}

function onConfirmDelete(e) {
  var d = deletableByKey_(e.parameters.key);
  var threads = GmailApp.search(deleteQuery_(d.label), 0, 500);
  var n = threads.length;
  var section = CardService.newCardSection();
  if (!n) {
    section.addWidget(CardService.newTextParagraph().setText('No conversations labelled ' + d.label + '.'));
  } else {
    section.addWidget(CardService.newTextParagraph().setText(
      'Move <b>' + (n >= 500 ? '500+' : n) + '</b> conversations labelled <b>' + escape_(d.label) + '</b> to Trash?<br>' +
      '<font color="#5f6368">Starred conversations are skipped. Trash keeps them for 30 days.</font>'));
    section.addWidget(CardService.newTextParagraph().setText(threads.slice(0, 8).map(function (t) {
      return '- ' + escape_((t.getFirstMessageSubject() || '(no subject)').slice(0, 70));
    }).join('<br>') + (n > 8 ? '<br>…' : '')));
    section.addWidget(CardService.newButtonSet()
      .addButton(CardService.newTextButton().setText('Move to Trash')
        .setTextButtonStyle(CardService.TextButtonStyle.FILLED)
        .setOnClickAction(CardService.newAction().setFunctionName('onDeleteGroup').setParameters({ key: d.key })))
      .addButton(CardService.newTextButton().setText('Review in Gmail')
        .setOpenLink(CardService.newOpenLink().setUrl(labelUrl_(d.label)))));
  }
  return navPush_(CardService.newCardBuilder()
    .setHeader(CardService.newCardHeader().setTitle('Delete: ' + d.name)).addSection(section).build());
}

function onDeleteGroup(e) {
  var d = deletableByKey_(e.parameters.key);
  var start = Date.now(), moved = 0, more = false;
  while (Date.now() - start < CARD_BUDGET_MS) {
    var batch = GmailApp.search(deleteQuery_(d.label), 0, 100);
    if (!batch.length) break;
    GmailApp.moveThreadsToTrash(batch);
    moved += batch.length;
    more = batch.length === 100;
    if (!more) break;
  }
  return navPush_(CardService.newCardBuilder()
    .setHeader(CardService.newCardHeader().setTitle('Moved ' + moved + ' to Trash'))
    .addSection(CardService.newCardSection()
      .addWidget(CardService.newTextParagraph().setText(
        'To undo, open Trash, select the conversations and choose "Move to Inbox".'))
      .addWidget(more
        ? CardService.newTextButton().setText('Continue deleting')
            .setOnClickAction(CardService.newAction().setFunctionName('onDeleteGroup').setParameters({ key: d.key }))
        : CardService.newTextButton().setText('Open Trash')
            .setOpenLink(CardService.newOpenLink().setUrl('https://mail.google.com/mail/u/0/#trash'))))
    .build());
}

// ---------------------------------------------------------------- single-thread actions

function onTrashThis(e) {
  GmailApp.setCurrentMessageAccessToken(e.gmail.accessToken);
  GmailApp.getMessageById(e.gmail.messageId).getThread().moveToTrash();
  return notify_('Moved to Trash');
}

/** "Wrong label?": relabel the thread and teach the backend a sender rule. */
function onCorrectLabel(e) {
  var topic = formValue_(e, 'correct_topic');
  if (!TOPICS[topic]) return notify_('Choose a topic first');
  var del = formValue_(e, 'correct_delete');          // 'auto' | 'yes' | 'no'
  var canDelete = del === 'yes' ? true : (del === 'no' ? false : null);
  var wholeDomain = formValue_(e, 'whole_domain') === 'yes';
  GmailApp.setCurrentMessageAccessToken(e.gmail.accessToken);
  var thread = GmailApp.getMessageById(e.gmail.messageId).getThread();
  var msgs = thread.getMessages();
  try {
    var res = callBackend_('post', '/triage/correct', {
      user_id: getUserId_(), sender: msgs[msgs.length - 1].getFrom(), predicted: e.parameters.predicted || '',
      topic: topic, can_delete: canDelete, whole_domain: wholeDomain
    });
    var labels = res.gmail_labels.slice();
    if (canDelete === null && e.parameters.priority === '1' && topic !== 'spam') labels.push(PRIORITY_LABEL);
    setThreadLabels_(thread, labels);
    CacheService.getUserCache().remove('an_' + e.gmail.messageId + '__0');
    return notify_('Saved. Mail from ' + res.sender_key + ' will be labelled ' + TOPICS[topic].name + '.');
  } catch (err) {
    return notify_('Could not save: ' + err.message);
  }
}

// ---------------------------------------------------------------- hourly auto-sort

function setAutoSort_(on) {
  ScriptApp.getProjectTriggers().forEach(function (t) {
    if (t.getHandlerFunction() === 'autoSortTrigger') ScriptApp.deleteTrigger(t);
  });
  if (on) ScriptApp.newTrigger('autoSortTrigger').timeBased().everyHours(1).create();
}

function isAutoSortOn_() {
  try {
    return ScriptApp.getProjectTriggers().some(function (t) { return t.getHandlerFunction() === 'autoSortTrigger'; });
  } catch (err) {
    return false;
  }
}

/** Runs hourly when enabled; sorts new mail from the last 2 days. */
function autoSortTrigger() {
  try {
    var r = sortInbox_(2, TRIGGER_BUDGET_MS);
    console.log('Auto-sort: ' + JSON.stringify(r.counts) + (r.error ? ' error: ' + r.error : ''));
  } catch (err) {
    console.error('Auto-sort failed: ' + err);
  }
}
