import 'dart:async';
import 'dart:js_interop';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:web/web.dart' as web;

import 'api.dart';

class ChatMessage {
  ChatMessage.user(this.text) : fromUser = true, isUpload = false;
  ChatMessage.assistant() : fromUser = false, isUpload = false, text = '';
  ChatMessage.upload(this.text) : fromUser = false, isUpload = true; // text = file name

  final bool fromUser;
  final bool isUpload;
  String text;
  List<Source> sources = [];
  Set<int> cited = {};
  List<int> bogus = [];
  String? error;
  bool done = false;
  String? status; // uploads: latest log line while indexing
  Map<String, dynamic>? report; // uploads: the ingest report
}

const _suggestions = [
  'What is the total in my expense report?',
  'Summarise the main points of my documents',
  'What word is written in my handwritten note?',
];

class ChatPage extends StatefulWidget {
  const ChatPage({super.key, required this.api});
  final LocalLensApi api;

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  final _messages = <ChatMessage>[];
  final _input = TextEditingController();
  final _scroll = ScrollController();
  late final FocusNode _inputFocus = FocusNode(onKeyEvent: _onKey);
  StreamSubscription<Map<String, dynamic>>? _answering;
  Health? _health;
  bool _serverDown = false;
  bool _uploading = false;
  bool _visionForAllScans = false; // same as `locallens ingest --vlm all-scans`

  bool get _busy => _answering != null || _uploading;

  @override
  void initState() {
    super.initState();
    _filePicker; // create it now, not on first click
    _refreshHealth();
  }

  @override
  void dispose() {
    _answering?.cancel();
    _filePicker.remove();
    _input.dispose();
    _scroll.dispose();
    _inputFocus.dispose();
    super.dispose();
  }

  Future<void> _refreshHealth() async {
    try {
      final h = await widget.api.health();
      if (mounted) {
        setState(() {
          _health = h;
          _serverDown = false;
        });
      }
    } catch (_) {
      if (mounted) setState(() => _serverDown = true);
    }
  }

  // Enter sends, Shift+Enter adds a new line.
  KeyEventResult _onKey(FocusNode node, KeyEvent e) {
    if (e is KeyDownEvent &&
        e.logicalKey == LogicalKeyboardKey.enter &&
        !HardwareKeyboard.instance.isShiftPressed) {
      _send();
      return KeyEventResult.handled;
    }
    return KeyEventResult.ignored;
  }

  void _send([String? preset]) {
    final q = (preset ?? _input.text).trim();
    if (q.isEmpty || _busy) return;
    final reply = ChatMessage.assistant();
    setState(() {
      _messages
        ..add(ChatMessage.user(q))
        ..add(reply);
      _input.clear();
    });
    _scrollToEnd();
    _answering = widget.api.ask(q).listen(
      (ev) => setState(() {
        switch (ev['type']) {
          case 'sources':
            reply.sources = [
              for (final s in ev['sources'] as List) Source.fromJson(s as Map<String, dynamic>)
            ];
          case 'token':
            reply.text += ev['text'] as String;
            _scrollToEnd();
          case 'done':
            reply.cited = {for (final n in ev['cited'] as List) n as int};
            reply.bogus = [for (final n in ev['bogus'] as List) n as int];
          case 'error':
            reply.error = ev['message'] as String;
        }
      }),
      onError: (Object e) => setState(() => reply.error = e.toString()),
      onDone: _finish(reply),
    );
  }

  void Function() _finish(ChatMessage reply) => () {
        if (!mounted) return;
        setState(() {
          reply.done = true;
          _answering = null;
        });
        _inputFocus.requestFocus();
        _refreshHealth();
      };

  void _stop() {
    _answering?.cancel();
    final reply = _messages.last;
    setState(() {
      reply.done = true;
      if (reply.text.isEmpty) reply.error = 'Stopped.';
      _answering = null;
    });
  }

  void _newChat() {
    _answering?.cancel();
    setState(() {
      _messages.clear();
      _answering = null;
    });
    _inputFocus.requestFocus();
  }

  // One hidden <input type="file"> kept in the page; Safari ignores clicks on detached inputs.
  late final web.HTMLInputElement _filePicker = _createFilePicker();

  web.HTMLInputElement _createFilePicker() {
    final input = web.HTMLInputElement()
      ..type = 'file'
      ..multiple = true
      ..accept = supportedExtensions.join(',')
      ..id = 'locallens-file-picker'
      ..setAttribute('aria-label', 'Add files to analyse');
    input.style.display = 'none';
    input.onchange = ((web.Event _) {
      final files = input.files;
      if (files == null || files.length == 0 || _busy) return;
      final picked = [for (var i = 0; i < files.length; i++) files.item(i)!];
      input.value = ''; // so picking the same file again still fires onchange
      _uploadAll(picked);
    }).toJS;
    web.document.body!.append(input);
    return input;
  }

  void _pickFiles() {
    if (!_busy) _filePicker.click();
  }

  Future<void> _uploadAll(List<web.File> files) async {
    setState(() => _uploading = true);
    for (final file in files) {
      final card = ChatMessage.upload(file.name);
      setState(() => _messages.add(card));
      _scrollToEnd();
      try {
        final vlm = _visionForAllScans ? 'all-scans' : 'handwriting';
        await for (final ev in widget.api.upload(file, vlm: vlm)) {
          setState(() {
            switch (ev['type']) {
              case 'saved':
                card.text = ev['name'] as String;
              case 'log':
                card.status = ev['text'] as String;
              case 'done':
                card.report = ev['report'] as Map<String, dynamic>;
              case 'error':
                card.error = ev['message'] as String;
            }
          });
        }
      } catch (e) {
        setState(() => card.error = e.toString());
      }
      if (!mounted) return;
      setState(() => card.done = true);
    }
    setState(() => _uploading = false);
    _refreshHealth();
    _inputFocus.requestFocus();
  }

  void _showDocuments() {
    showDialog<void>(context: context, builder: (_) => _DocumentsDialog(api: widget.api));
  }

  void _scrollToEnd() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scroll.hasClients) {
        _scroll.animateTo(_scroll.position.maxScrollExtent,
            duration: const Duration(milliseconds: 200), curve: Curves.easeOut);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            _TopBar(
              health: _health,
              serverDown: _serverDown,
              onNewChat: _newChat,
              onDocuments: _showDocuments,
              visionForAllScans: _visionForAllScans,
              onVisionChanged: (v) => setState(() => _visionForAllScans = v),
            ),
            Expanded(
              child: _messages.isEmpty
                  ? _Welcome(onPick: _send, onUpload: _busy ? null : _pickFiles)
                  : ListView.builder(
                      controller: _scroll,
                      padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
                      itemCount: _messages.length,
                      itemBuilder: (_, i) {
                        final m = _messages[i];
                        return _Centered(
                          child: m.fromUser
                              ? _UserBubble(text: m.text)
                              : m.isUpload
                                  ? _UploadCard(msg: m)
                                  : _AssistantMessage(msg: m),
                        );
                      },
                    ),
            ),
            _Composer(
              controller: _input,
              focusNode: _inputFocus,
              answering: _answering != null,
              uploading: _uploading,
              onSend: _send,
              onStop: _stop,
              onAttach: _pickFiles,
            ),
          ],
        ),
      ),
    );
  }
}

class _Centered extends StatelessWidget {
  const _Centered({required this.child});
  final Widget child;

  @override
  Widget build(BuildContext context) => Center(
        child: ConstrainedBox(constraints: const BoxConstraints(maxWidth: 760), child: child),
      );
}

class _TopBar extends StatelessWidget {
  const _TopBar({
    required this.health,
    required this.serverDown,
    required this.onNewChat,
    required this.onDocuments,
    required this.visionForAllScans,
    required this.onVisionChanged,
  });
  final Health? health;
  final bool serverDown;
  final VoidCallback onNewChat;
  final VoidCallback onDocuments;
  final bool visionForAllScans;
  final ValueChanged<bool> onVisionChanged;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final (color, label) = serverDown
        ? (cs.error, 'LocalLens server not reachable')
        : health == null
            ? (cs.outline, 'Connecting…')
            : (Colors.green, 'Offline · ${health!.documents} documents · ${health!.chunks} chunks');
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      decoration: BoxDecoration(border: Border(bottom: BorderSide(color: cs.outlineVariant))),
      child: Row(
        children: [
          Icon(Icons.center_focus_strong_rounded, color: cs.primary),
          const SizedBox(width: 8),
          Text('LocalLens', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(width: 12),
          Expanded(
            child: Row(children: [
              Container(
                  width: 8,
                  height: 8,
                  decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
              const SizedBox(width: 6),
              Flexible(
                child: Text(label,
                    overflow: TextOverflow.ellipsis,
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(color: cs.onSurfaceVariant)),
              ),
            ]),
          ),
          IconButton(
            tooltip: 'Indexed documents',
            onPressed: onDocuments,
            icon: const Icon(Icons.folder_open_outlined),
          ),
          PopupMenuButton<bool>(
            tooltip: 'Upload settings',
            icon: const Icon(Icons.tune),
            onSelected: onVisionChanged,
            itemBuilder: (_) => [
              CheckedPopupMenuItem(
                value: !visionForAllScans,
                checked: visionForAllScans,
                child: const ListTile(
                  contentPadding: EdgeInsets.zero,
                  title: Text('Read every scan with the vision model'),
                  subtitle: Text('Best for handwriting and scanned tables. Slower: about a minute per page.'),
                ),
              ),
            ],
          ),
          IconButton(
            tooltip: 'New chat',
            onPressed: onNewChat,
            icon: const Icon(Icons.edit_square),
          ),
        ],
      ),
    );
  }
}

class _Welcome extends StatelessWidget {
  const _Welcome({required this.onPick, required this.onUpload});
  final void Function(String) onPick;
  final VoidCallback? onUpload;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context).textTheme;
    final cs = Theme.of(context).colorScheme;
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: _Centered(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.center_focus_strong_rounded, size: 44, color: cs.primary),
              const SizedBox(height: 16),
              Text('Ask about your documents', style: t.headlineSmall, textAlign: TextAlign.center),
              const SizedBox(height: 8),
              Text(
                'Answers come only from the files you indexed, with page citations. '
                'Nothing leaves this computer.',
                style: t.bodyMedium?.copyWith(color: cs.onSurfaceVariant),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 20),
              FilledButton.tonalIcon(
                onPressed: onUpload,
                icon: const Icon(Icons.upload_file),
                label: const Text('Add files to analyse'),
              ),
              const SizedBox(height: 6),
              Text('PDF, PNG, JPG, TIFF, BMP or GIF. Scans, photos and handwriting work too.',
                  style: t.bodySmall?.copyWith(color: cs.onSurfaceVariant), textAlign: TextAlign.center),
              const SizedBox(height: 24),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                alignment: WrapAlignment.center,
                children: [
                  for (final s in _suggestions)
                    ActionChip(label: Text(s), onPressed: () => onPick(s)),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _UserBubble extends StatelessWidget {
  const _UserBubble({required this.text});
  final String text;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Align(
      alignment: Alignment.centerRight,
      child: Container(
        margin: const EdgeInsets.only(top: 8, bottom: 16, left: 64),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        decoration: BoxDecoration(
          color: cs.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(20),
        ),
        child: SelectableText(text, style: Theme.of(context).textTheme.bodyLarge),
      ),
    );
  }
}

class _AssistantMessage extends StatelessWidget {
  const _AssistantMessage({required this.msg});
  final ChatMessage msg;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final t = Theme.of(context).textTheme;
    final waiting = !msg.done && msg.text.isEmpty && msg.error == null;
    return Padding(
      padding: const EdgeInsets.only(bottom: 24),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          CircleAvatar(
            radius: 14,
            backgroundColor: cs.primaryContainer,
            child: Icon(Icons.center_focus_strong_rounded, size: 16, color: cs.onPrimaryContainer),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (waiting)
                  Row(children: [
                    const SizedBox(
                        width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2)),
                    const SizedBox(width: 10),
                    Text(msg.sources.isEmpty ? 'Searching your documents…' : 'Writing the answer…',
                        style: t.bodyMedium?.copyWith(color: cs.onSurfaceVariant)),
                  ]),
                if (msg.text.isNotEmpty)
                  SelectableText.rich(_answerSpan(msg.text.trim(), msg, cs, t.bodyLarge!)),
                if (msg.error != null) ...[
                  const SizedBox(height: 8),
                  _Notice(icon: Icons.error_outline, color: cs.error, text: msg.error!),
                ],
                if (msg.done && msg.bogus.isNotEmpty)
                  _Notice(
                      icon: Icons.warning_amber_rounded,
                      color: Colors.orange.shade800,
                      text: 'The answer cites sources that do not exist: '
                          '${msg.bogus.map((n) => '[$n]').join(' ')}'),
                if (msg.done && msg.text.isNotEmpty && msg.cited.isEmpty && msg.error == null)
                  _Notice(
                      icon: Icons.info_outline,
                      color: cs.onSurfaceVariant,
                      text: 'This answer did not cite a source. Check the sources below.'),
                if (msg.sources.isNotEmpty && (msg.done || msg.text.isNotEmpty))
                  _SourcesPanel(msg: msg),
              ],
            ),
          ),
        ],
      ),
    );
  }

  /// Plain text with **bold** and [n] citation markers styled.
  static TextSpan _answerSpan(String text, ChatMessage msg, ColorScheme cs, TextStyle base) {
    final spans = <InlineSpan>[];
    final pattern = RegExp(r'\*\*(.+?)\*\*|\[(\d+)\]');
    var last = 0;
    for (final m in pattern.allMatches(text)) {
      if (m.start > last) spans.add(TextSpan(text: text.substring(last, m.start)));
      if (m.group(1) != null) {
        spans.add(TextSpan(text: m.group(1), style: const TextStyle(fontWeight: FontWeight.w600)));
      } else {
        final n = int.parse(m.group(2)!);
        final real = msg.sources.any((s) => s.n == n);
        spans.add(TextSpan(
          text: '[$n]',
          style: TextStyle(
            color: real ? cs.primary : cs.error,
            fontWeight: FontWeight.w600,
            fontSize: (base.fontSize ?? 16) * 0.85,
          ),
        ));
      }
      last = m.end;
    }
    if (last < text.length) spans.add(TextSpan(text: text.substring(last)));
    return TextSpan(style: base.copyWith(height: 1.55), children: spans);
  }
}

class _UploadCard extends StatelessWidget {
  const _UploadCard({required this.msg});
  final ChatMessage msg;

  static int _n(Map<String, dynamic> r, String key) => (r[key] as int?) ?? 0;

  static String _summary(Map<String, dynamic> r) {
    if (_n(r, 'files_indexed') == 0 && _n(r, 'files_skipped_unchanged') > 0) {
      return 'Already in your index, so there was nothing new to add.';
    }
    final pages = _n(r, 'pages_digital') + _n(r, 'pages_ocr');
    final parts = [
      '$pages page${pages == 1 ? '' : 's'}',
      if (_n(r, 'pages_ocr') > 0) '${_n(r, 'pages_ocr')} read from images',
      if (_n(r, 'pages_vlm') > 0) '${_n(r, 'pages_vlm')} by the vision model',
      if (_n(r, 'tables') > 0) '${_n(r, 'tables')} tables',
      '${_n(r, 'chunks')} chunks',
    ];
    return '${parts.join(' · ')}. You can ask about it now.';
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final t = Theme.of(context).textTheme;
    final report = msg.report;
    final errors = [...?(report?['errors'] as List?)?.cast<String>(), ?msg.error];
    final failed = msg.done && (errors.isNotEmpty || report == null);
    final (Widget icon, String title) = !msg.done
        ? (const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2)), 'Reading')
        : failed
            ? (Icon(Icons.error_outline, color: cs.error), 'Could not add')
            : (const Icon(Icons.check_circle, color: Colors.green), 'Added');
    final detail = failed
        ? (errors.isEmpty ? 'The upload stopped before indexing finished.' : errors.join('\n'))
        : msg.done
            ? _summary(report!)
            : (msg.status ?? 'Uploading and indexing. Scanned pages can take a minute each.');
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        border: Border.all(color: failed ? cs.error : cs.outlineVariant),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          icon,
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('$title ${msg.text}', style: t.labelLarge),
                const SizedBox(height: 2),
                Text(detail,
                    style: t.bodySmall?.copyWith(color: failed ? cs.error : cs.onSurfaceVariant)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _DocumentsDialog extends StatelessWidget {
  const _DocumentsDialog({required this.api});
  final LocalLensApi api;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context).textTheme;
    final cs = Theme.of(context).colorScheme;
    return AlertDialog(
      title: const Text('Indexed documents'),
      content: SizedBox(
        width: 520,
        child: FutureBuilder<List<IndexedDocument>>(
          future: api.documents(),
          builder: (context, snap) {
            if (snap.hasError) return Text('Could not load the list: ${snap.error}');
            if (!snap.hasData) {
              return const SizedBox(height: 80, child: Center(child: CircularProgressIndicator()));
            }
            final docs = snap.data!;
            if (docs.isEmpty) return const Text('Nothing indexed yet. Add files with the paperclip button.');
            return ListView.separated(
              shrinkWrap: true,
              itemCount: docs.length,
              separatorBuilder: (_, _) => const Divider(height: 1),
              itemBuilder: (_, i) {
                final d = docs[i];
                final scanned = d.pages - d.digitalPages;
                return ListTile(
                  contentPadding: EdgeInsets.zero,
                  leading: Icon(scanned > 0 ? Icons.image_outlined : Icons.description_outlined),
                  title: Text(d.name, style: t.bodyMedium),
                  subtitle: Text(
                    '${d.pages} page${d.pages == 1 ? '' : 's'}'
                    '${scanned > 0 ? ' ($scanned scanned)' : ''} · ${d.chunks} chunks · '
                    'added ${d.indexed.replaceFirst('T', ' ').substring(0, 16)}',
                    style: t.bodySmall?.copyWith(color: cs.onSurfaceVariant),
                  ),
                );
              },
            );
          },
        ),
      ),
      actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text('Close'))],
    );
  }
}

class _Notice extends StatelessWidget {
  const _Notice({required this.icon, required this.color, required this.text});
  final IconData icon;
  final Color color;
  final String text;

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(top: 8),
        child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Icon(icon, size: 16, color: color),
          const SizedBox(width: 6),
          Expanded(
              child: Text(text, style: Theme.of(context).textTheme.bodySmall?.copyWith(color: color))),
        ]),
      );
}

class _SourcesPanel extends StatelessWidget {
  const _SourcesPanel({required this.msg});
  final ChatMessage msg;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final t = Theme.of(context).textTheme;
    // cited sources first, then the rest in rank order
    final ordered = [
      ...msg.sources.where((s) => msg.cited.contains(s.n)),
      ...msg.sources.where((s) => !msg.cited.contains(s.n)),
    ];
    final citedCount = msg.sources.where((s) => msg.cited.contains(s.n)).length;
    return Container(
      margin: const EdgeInsets.only(top: 12),
      decoration: BoxDecoration(
        border: Border.all(color: cs.outlineVariant),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          dense: true,
          shape: const RoundedRectangleBorder(),
          title: Text(
            'Sources (${msg.sources.length})${citedCount > 0 ? ' · $citedCount cited' : ''}',
            style: t.labelLarge,
          ),
          children: [
            for (final s in ordered)
              _SourceRow(source: s, cited: msg.cited.contains(s.n)),
            const SizedBox(height: 8),
          ],
        ),
      ),
    );
  }
}

class _SourceRow extends StatelessWidget {
  const _SourceRow({required this.source, required this.cited});
  final Source source;
  final bool cited;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final t = Theme.of(context).textTheme;
    final how = switch (source.origin) {
      'ocr' => 'OCR',
      'vlm' => 'vision model',
      _ => null,
    };
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 4, 16, 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 24,
            height: 24,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: cited ? cs.primary : Colors.transparent,
              border: Border.all(color: cited ? cs.primary : cs.outline),
              shape: BoxShape.circle,
            ),
            child: Text('${source.n}',
                style: t.labelSmall?.copyWith(color: cited ? cs.onPrimary : cs.onSurfaceVariant)),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Wrap(spacing: 6, crossAxisAlignment: WrapCrossAlignment.center, children: [
                  Text(source.document, style: t.labelLarge),
                  Text('page ${source.page}', style: t.bodySmall?.copyWith(color: cs.onSurfaceVariant)),
                  if (how != null)
                    Text('· $how${source.confidence != null ? ' ${source.confidence!.toStringAsFixed(2)}' : ''}',
                        style: t.bodySmall?.copyWith(
                            color: source.lowConfidence ? Colors.orange.shade800 : cs.onSurfaceVariant)),
                  if (source.lowConfidence)
                    Text('low confidence',
                        style: t.bodySmall?.copyWith(color: Colors.orange.shade800)),
                ]),
                const SizedBox(height: 2),
                SelectableText(source.snippet,
                    style: t.bodySmall?.copyWith(color: cs.onSurfaceVariant, height: 1.4)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _Composer extends StatelessWidget {
  const _Composer({
    required this.controller,
    required this.focusNode,
    required this.answering,
    required this.uploading,
    required this.onSend,
    required this.onStop,
    required this.onAttach,
  });
  final TextEditingController controller;
  final FocusNode focusNode;
  final bool answering;
  final bool uploading;
  final void Function([String?]) onSend;
  final VoidCallback onStop;
  final VoidCallback onAttach;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
      child: _Centered(
        child: Column(
          children: [
            Container(
              padding: const EdgeInsets.fromLTRB(4, 4, 6, 4),
              decoration: BoxDecoration(
                color: cs.surfaceContainerHigh,
                borderRadius: BorderRadius.circular(26),
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Padding(
                    padding: const EdgeInsets.only(bottom: 4),
                    child: IconButton(
                      tooltip: 'Add files to analyse',
                      onPressed: answering || uploading ? null : onAttach,
                      icon: const Icon(Icons.attach_file),
                    ),
                  ),
                  Expanded(
                    child: TextField(
                      controller: controller,
                      focusNode: focusNode,
                      autofocus: true,
                      minLines: 1,
                      maxLines: 6,
                      keyboardType: TextInputType.multiline,
                      decoration: const InputDecoration(
                        hintText: 'Ask a question about your documents',
                        border: InputBorder.none,
                        contentPadding: EdgeInsets.symmetric(vertical: 12),
                      ),
                    ),
                  ),
                  const SizedBox(width: 6),
                  Padding(
                    padding: const EdgeInsets.only(bottom: 4),
                    child: answering
                        ? IconButton.filledTonal(
                            tooltip: 'Stop',
                            onPressed: onStop,
                            icon: const Icon(Icons.stop_rounded))
                        : ListenableBuilder(
                            listenable: controller,
                            builder: (_, _) => IconButton.filled(
                              tooltip: 'Send',
                              onPressed:
                                  uploading || controller.text.trim().isEmpty ? null : () => onSend(),
                              icon: const Icon(Icons.arrow_upward_rounded),
                            ),
                          ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 6),
            Text(
              'Runs entirely on this computer. Answers can be wrong, so check the sources.',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(color: cs.onSurfaceVariant),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}
