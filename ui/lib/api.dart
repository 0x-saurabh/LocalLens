// Talks to the LocalLens server that serves this page (same origin, 127.0.0.1).
import 'dart:async';
import 'dart:convert';
import 'dart:js_interop';

import 'package:web/web.dart' as web;

class ApiException implements Exception {
  ApiException(this.message);
  final String message;
  @override
  String toString() => message;
}

class Source {
  Source.fromJson(Map<String, dynamic> j)
      : n = j['n'] as int,
        document = j['document'] as String,
        page = j['page'] as int,
        snippet = j['snippet'] as String,
        origin = j['source'] as String,
        confidence = (j['confidence'] as num?)?.toDouble();

  final int n;
  final String document;
  final int page;
  final String snippet;
  final String origin; // textlayer | ocr | vlm
  final double? confidence;

  // Same threshold the answer model is told about in generate.py.
  bool get lowConfidence => confidence != null && confidence! < 0.8;
}

class Health {
  Health(this.documents, this.chunks);
  final int documents;
  final int chunks;
}

class IndexedDocument {
  IndexedDocument.fromJson(Map<String, dynamic> j)
      : name = j['name'] as String,
        pages = j['pages'] as int,
        digitalPages = j['digital_pages'] as int,
        chunks = j['chunks'] as int,
        indexed = j['indexed'] as String;

  final String name;
  final int pages;
  final int digitalPages;
  final int chunks;
  final String indexed;
}

/// File types the server accepts (ingest/discover.py SUPPORTED).
const supportedExtensions = ['.pdf', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'];

class LocalLensApi {
  /// Streams the server's newline-delimited JSON events for one question.
  /// Cancelling the subscription aborts the request.
  Stream<Map<String, dynamic>> ask(String question) =>
      _ndjson('/api/ask', 'application/json', jsonEncode({'question': question}).toJS);

  /// Uploads one file and streams the indexing events ("saved", "log", "done" or "error").
  /// [vlm] is handwriting | all-scans | off, like `locallens ingest --vlm`.
  Stream<Map<String, dynamic>> upload(web.File file, {String vlm = 'handwriting'}) => _ndjson(
      '/api/upload?name=${Uri.encodeQueryComponent(file.name)}&vlm=$vlm',
      'application/octet-stream',
      file);

  Future<List<IndexedDocument>> documents() async {
    final list = jsonDecode(await _get('/api/documents')) as List;
    return [for (final d in list) IndexedDocument.fromJson(d as Map<String, dynamic>)];
  }

  Future<Health> health() async {
    final j = jsonDecode(await _get('/api/health')) as Map<String, dynamic>;
    return Health(j['documents'] as int, j['chunks'] as int);
  }

  Future<String> _get(String path) async {
    final resp = await web.window.fetch(path.toJS).toDart;
    if (!resp.ok) throw ApiException('Server answered ${resp.status}');
    return (await resp.text().toDart).toDart;
  }

  // XHR rather than fetch: its progress events expose the partial body, so each JSON line can be
  // handled as soon as it arrives.
  Stream<Map<String, dynamic>> _ndjson(String url, String contentType, JSAny body) {
    final xhr = web.XMLHttpRequest();
    late final StreamController<Map<String, dynamic>> out;
    var consumed = 0;
    var pending = '';

    void drain() {
      final text = xhr.responseText;
      pending += text.substring(consumed);
      consumed = text.length;
      int nl;
      while ((nl = pending.indexOf('\n')) >= 0) {
        final line = pending.substring(0, nl).trim();
        pending = pending.substring(nl + 1);
        if (line.isNotEmpty) out.add(jsonDecode(line) as Map<String, dynamic>);
      }
    }

    out = StreamController<Map<String, dynamic>>(onCancel: () => xhr.abort());
    xhr.open('POST', url);
    xhr.setRequestHeader('Content-Type', contentType);
    xhr.onprogress = ((web.Event _) {
      if (xhr.status == 200) drain();
    }).toJS;
    xhr.onload = ((web.Event _) {
      if (xhr.status == 200) {
        drain();
      } else {
        out.addError(ApiException(_errorText(xhr.responseText, xhr.status)));
      }
      out.close();
    }).toJS;
    xhr.onerror = ((web.Event _) {
      out.addError(ApiException(
          'Cannot reach LocalLens. Is `locallens ui` still running in the terminal?'));
      out.close();
    }).toJS;
    xhr.send(body);
    return out.stream;
  }

  static String _errorText(String body, int status) {
    try {
      return (jsonDecode(body) as Map<String, dynamic>)['error'] as String;
    } catch (_) {
      return 'Request failed ($status)';
    }
  }
}
